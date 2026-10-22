"""Grace tools server: the decision checker (JevK5 4B on the Arc iGPU) and every tool Grace can
call: web search with a lookup loop, opening programs and web pages, and finding, listing and
opening files (ADR 0019, ADR 0020).

  bench/.venv-openvino/Scripts/python.exe services/tools_server.py [--dry-run]

Localhost TCP 127.0.0.1:7813, newline-delimited JSON, one request per line:
  {"kind": "manifest"}                       -> {"tools": [<OpenAI tool definitions>], "slow": [...]}
  {"kind": "decide", "state", "question"}    -> {"answer": {...}}       (JevK5 typed decision)
  {"kind": "tool", "name", "args"}           -> {"ok": true, "result": "<text for Gemma>", ...}

Tools that act on the PC (open_app, open_url, open_file) only start programs or open pages and
documents; nothing is deleted, sent or bought. --dry-run (benches) reports what they would do.
Search results and file contents are data for Gemma, never instructions.
"""

import argparse
import difflib
import html
import json
import os
import re
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import openvino as ov
from jevk5.prompt import LETTERS, answer, decision_options, prompt_text
from transformers import AutoTokenizer

import grace_tools
from grace_tools import registry

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / ".runtime/models/jevk5-ov/4b-int8-stateless"
DEVICE = "GPU.0"  # the Arc iGPU: keeps the RTX for Gemma (decode 50 -> 48.5 tok/s, vs 33 on the RTX)
TEMPERATURE = 1.22  # JevK5 v0.3 4B's calibration temperature (model card)
PORT = 7813
SEARXNG = "http://127.0.0.1:8888"
LLM = "http://127.0.0.1:8080"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

# The lookup loop. A fact must be stated outright; an overview (plot, what something is) needs
# relevant information; news is read from the news sites' front pages.
SURE = {"fact": 0.85, "overview": 0.7, "live": 0.6}
# Live values (weather, opening hours, prices) can't be confirmed further by searching again.
ROUNDS = {"fact": 3, "overview": 2, "live": 1}
GOOD_ENOUGH = 0.7  # a fact passage this sure after reading pages ends the loop too
MAX_ROUNDS = 3
BUDGET_S = 12.0
SNIPPETS = 6
PAGES = 3
PASSAGES_PER_PAGE = 2

HOME = Path.home()
FILE_ROOTS = [HOME / d for d in ("Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music")]
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "target", ".runtime", "dist",
             "build", ".cache", "bench-results", "traces", ".next", ".idea", ".vs", "site-packages",
             "$recycle.bin", "appdata", ".gradle", ".m2", "obj", "bin", ".claude"}
MAX_FILES = 400_000
NO_OPEN = {".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".msi", ".com", ".scr", ".lnk", ".reg",
           ".jar", ".py", ".psm1", ".hta", ".wsf", ".cpl", ".dll", ".sys"}
KIND_EXT = {
    "pdf": {".pdf"}, "word": {".docx", ".doc"}, "dokument": {".docx", ".doc", ".pdf", ".md", ".txt", ".odt"},
    "powerpoint": {".pptx", ".ppt"}, "præsentation": {".pptx", ".ppt", ".pdf"}, "slides": {".pptx", ".ppt"},
    "excel": {".xlsx", ".xls", ".csv"}, "regneark": {".xlsx", ".xls", ".csv"},
    "billede": {".jpg", ".jpeg", ".png", ".heic", ".webp"}, "foto": {".jpg", ".jpeg", ".png", ".heic"},
    "video": {".mp4", ".mov", ".mkv"}, "noter": {".md", ".txt", ".docx"}, "zip": {".zip", ".7z", ".rar"},
}
# Words Mikkel says vs words in file names.
SYNONYMS = {"cv": "resume curriculum", "ansøgning": "application", "opgave": "assignment",
            "eksamen": "exam", "rapport": "report", "noter": "notes", "billede": "img",
            "vejleder": "vejledermøde supervisor"}
FOLDER_ALIASES = {
    "skrivebord": HOME / "Desktop", "skrivebordet": HOME / "Desktop", "desktop": HOME / "Desktop",
    "dokumenter": HOME / "Documents", "documents": HOME / "Documents", "overførsler": HOME / "Downloads",
    "downloads": HOME / "Downloads", "billeder": HOME / "Pictures", "pictures": HOME / "Pictures",
    "videoer": HOME / "Videos", "musik": HOME / "Music",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ---- the checker ------------------------------------------------------------------------------

class Superseded(Exception):
    """A newer request with the same key arrived while this one waited for the checker."""


class Checker:
    """JevK5 4B through OpenVINO: one prefill per decision, the option letters' logits read and
    renormalised under the model's temperature (the same readout as jevk5's llama.cpp client)."""

    def __init__(self):
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(MODEL / "openvino_model.xml"), DEVICE)
        self.names = [i.get_any_name() for i in self.compiled.inputs]
        self.req = self.compiled.create_infer_request()
        self.letter_ids = [self.tok.convert_tokens_to_ids(c) for c in LETTERS]
        self.lock = threading.Lock()
        log(f"checker: JevK5 4B on {DEVICE} in {time.perf_counter() - t0:.1f} s")
        self.decide("The sky is blue.", {"type": "noul", "instructions": "Is the sky blue?"})

    def decide(self, state, question, wanted=lambda: True):
        options = decision_options(question)
        if len(options) > len(LETTERS):
            raise ValueError("more than 16 options is not supported here")
        texts = [t for _, t in options]
        ids = self.tok(prompt_text(state, question["instructions"], texts), add_special_tokens=False)["input_ids"][-6000:]
        x = np.array([ids], dtype=np.int64)
        feed = {"input_ids": x, "attention_mask": np.ones_like(x)}
        if "position_ids" in self.names:
            feed["position_ids"] = np.arange(x.shape[1], dtype=np.int64)[None, :]
        with self.lock:
            if not wanted():
                raise Superseded()
            row = self.req.infer(feed)[self.compiled.outputs[0]][0, -1].astype(np.float64)
        z = np.array([row[self.letter_ids[i]] for i in range(len(texts))]) / TEMPERATURE
        p = np.exp(z - z.max())
        p = p / p.sum()
        return answer(question, {k: float(v) for (k, _), v in zip(options, p)}, len(ids))

    def yes(self, state, instructions):
        return self.decide(state, {"type": "noul", "instructions": instructions})["noul"]


# ---- the web ----------------------------------------------------------------------------------

def get(url, timeout, limit=2_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "da,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(limit)
        charset = r.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def strip_tags(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def text_of(page):
    page = re.sub(r"(?is)<(script|style|noscript|svg|head|nav|footer|form)\b.*?</\1>", " ", page)
    page = re.sub(r"(?s)<!--.*?-->", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", page)
    page = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in page.split("\n"))
    return "\n".join(line for line in lines if len(line) > 30)


def headlines(page, base, n=10):
    """A news front page's article links: (headline, absolute url), in page order. Live run 7:
    without the links Grace made up an article URL (404) when asked to open the story."""
    page = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer)\b.*?</\1>", " ", page)
    seen, out = set(), []
    for href, inner in re.findall(r'(?is)<a\b[^>]*href="([^"#]+)"[^>]*>(.*?)</a>', page):
        t = strip_tags(inner)
        if not (25 <= len(t) <= 160) or t.lower() in seen:
            continue
        if re.search(r"cookie|abonn|log ind|nyhedsbrev|privatliv|kontakt|annonce", t, re.I):
            continue
        url = urllib.parse.urljoin(base, html.unescape(href))
        if urllib.parse.urlsplit(url).netloc != urllib.parse.urlsplit(base).netloc:
            continue
        seen.add(t.lower())
        out.append((t, url))
        if len(out) >= n:
            break
    return out


def published(page):
    """An article's publication time from its metadata, as 'YYYY-MM-DD HH:MM', or None."""
    m = (re.search(r'(?i)<meta[^>]+(?:property|name)="(?:article:published_time|pubdate|date)"[^>]+content="([^"]+)"', page)
         or re.search(r'"datePublished"\s*:\s*"([^"]+)"', page))
    if not m:
        return None
    return m.group(1).replace("T", " ")[:16]


SEARCH_CACHE = {}  # (query, language) -> (time, hits, engine)
CACHE_S = 600


def search(query, language="all"):
    """[(title, url, snippet)], cached for 10 minutes: the same search isn't sent twice (search
    engines suspended SearXNG for "too many requests" after a benchmark, 2026-10-21)."""
    key = (query.strip().lower(), language)
    hit = SEARCH_CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_S:
        return hit[1], hit[2] + "-cached"
    hits, engine = _search(query, language)
    if hits:
        if len(SEARCH_CACHE) > 500:
            SEARCH_CACHE.clear()
        SEARCH_CACHE[key] = (time.time(), hits, engine)
    return hits, engine


def _search(query, language="all"):
    """[(title, url, snippet)] from SearXNG, else DuckDuckGo's HTML page, else Wikipedia."""
    try:
        d = json.loads(get(f"{SEARXNG}/search?" + urllib.parse.urlencode({"q": query, "format": "json", "language": language}), 8))
        hits = [(r.get("title", ""), r["url"], r.get("content") or "") for r in d.get("results", []) if r.get("url")]
        if hits:
            return hits, "searxng"
    except Exception as e:
        log("searxng failed:", e)
    try:
        page = get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}), 8)
        hits = []
        for m in re.finditer(r'(?s)class="result__a" href="([^"]+)".*?>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>', page):
            url = m.group(1)
            if "uddg=" in url:
                url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
            hits.append((strip_tags(m.group(2)), url, strip_tags(m.group(3))))
        if hits:
            return hits, "duckduckgo"
    except Exception as e:
        log("duckduckgo failed:", e)
    try:
        d = json.loads(get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5}), 8))
        return [(r["title"], "https://en.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
                 strip_tags(r["snippet"])) for r in d["query"]["search"]], "wikipedia"
    except Exception as e:
        log("wikipedia failed:", e)
    return [], "none"


def words(s):
    return set(re.findall(r"[a-zæøå0-9]{3,}", s.lower()))


def passages(text, keys, n):
    paras, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > 700 and cur:
            paras.append(cur)
            cur = ""
        cur += line + " "
    if cur:
        paras.append(cur)
    ranked = sorted(paras, key=lambda p: -len(words(p) & keys))
    return [p.strip()[:900] for p in ranked[:n] if words(p) & keys]


def gemma(prompt, max_tokens=40):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(f"{LLM}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())["choices"][0]["message"]["content"].strip()


def page_text(url, timeout):
    try:
        return get(url, timeout)
    except Exception as e:
        log("fetch failed:", url, e)
        return ""


def story_time(when):
    """'2026-10-21 07:00' (or with seconds/zone) -> epoch seconds, or None."""
    try:
        return datetime.strptime(when[:16], "%Y-%m-%d %H:%M").timestamp()
    except (TypeError, ValueError):
        return None


def news(checker, question, query, language, alive):
    """Stories from the news sites a search finds, read from their front pages now, newest first,
    each with its publication time (live run 7: a September verdict was told as news from the last
    24 hours). When he asks for recent news, stories older than two days are left out."""
    t0 = time.perf_counter()
    recent = bool(re.search(r"døgn|i dag|seneste|sidste|last|latest|today|recent|24", question + " " + query, re.I))
    stories = []  # (epoch or None, headline, link, site, date text)
    # The sites a search turns up vary (tools sim: "nyheder Odense sidste 24 timer" found only sites
    # whose front pages gave no dated stories), so read more of them, and if that gives nothing,
    # search again without the time words.
    simple = re.sub(r"(?i)\b(sidste|seneste|døgn|i dag|24|timer|last|latest|hours?|today|news|nyheder)\b", " ", query)
    def enough():
        # When he asks for recent news, only stories from the last two days count.
        return sum(1 for st in stories if st[0] and (not recent or time.time() - st[0] < 2 * 86400))

    for q in dict.fromkeys([query, f"{' '.join(simple.split())} nyheder".strip()]):
        hits, engine = search(q, language)
        seen_sites = {st[3] for st in stories}
        sites = [h for h in hits if not re.search(r"wikipedia|reddit|facebook|youtube|weather|vejr|tutiempo", h[1], re.I)
                 and urllib.parse.urlsplit(h[1]).netloc not in seen_sites][:6]
        log(f"news [{engine}] {q!r}: reading {[urllib.parse.urlsplit(h[1]).netloc for h in sites]}")
        with ThreadPoolExecutor(6) as pool:
            pages = list(pool.map(lambda h: page_text(h[1], 6), sites))
            for (title, url, _), page in zip(sites, pages):
                if not alive():
                    break
                hs = headlines(page, url)[:6]
                if len(hs) < 3:
                    continue
                dates = list(pool.map(lambda h: published(page_text(h[1], 4)), hs))
                site = urllib.parse.urlsplit(url).netloc
                stories += [(story_time(d), head, link, site, d) for (head, link), d in zip(hs, dates)]
                if enough() >= 6:
                    break
        if enough() >= 3 or not alive():
            break
    now = time.time()
    stale = ""
    if recent:
        fresh = [st for st in stories if st[0] is None or now - st[0] < 2 * 86400]
        if not fresh and stories:
            stale = " None is from the last two days; these are the newest found."
        else:
            stories = fresh
    stories.sort(key=lambda st: -(st[0] or 0))
    lines = [f"- {head} [{d or 'date unknown'}] ({site}) {link}" for _, head, link, site, d in stories[:8]]
    evidence = []
    if lines:
        dated = sum(1 for st in stories[:8] if st[0])
        evidence.append(dict(title="News", url=sites[0][1] if sites else "", p=1.0,
                             text=(f"Stories from the news sites' front pages right now, newest first ({dated} with a "
                                   f"publication time; current time {datetime.now():%Y-%m-%d %H:%M}).{stale} Say when a story is "
                                   f"from; never call one recent if its date is unknown:\n") + "\n".join(lines)))
    status = "answered" if evidence else "not_found"
    return dict(status=status, kind="news", evidence=evidence, queries=[query], rounds=1, decisions=0,
                ms=round((time.perf_counter() - t0) * 1000, 1))


def lookup(checker, question, query, kind, language, alive):
    if kind == "news":
        return news(checker, question, query, language, alive)
    t0 = time.perf_counter()
    left = lambda: BUDGET_S - (time.perf_counter() - t0)
    # Tools sim v1: "Runner (2026 film)" passed as the plot of Runner Runner, so an overview must be
    # about the exact title. For facts that clause cost 1 of 18 right answers (lookup bench v2).
    same = " It must be about exactly that title, not something similarly named."
    criterion = {
        "fact": f"Does the evidence directly state the answer to this exact question: {question}",
        "overview": f"Does the evidence give the information this question asks for: {question}{same}",
        "live": f"Does the evidence give the current information this question asks for, for the right place or thing: {question}",
    }[kind]
    sure = SURE.get(kind, SURE["fact"])
    queries, tried, found, decisions = [], set(), [], 0
    keys = words(question + " " + query)
    pool = ThreadPoolExecutor(PAGES)
    status = "not_found"
    rounds = ROUNDS.get(kind, MAX_ROUNDS)
    for rnd in range(1, rounds + 1):
        if not alive():
            status = "cancelled"
            break
        queries.append(query)
        hits, engine = search(query, language)
        hits = [h for h in hits if h[1] not in tried][:SNIPPETS]
        log(f"round {rnd} [{engine}] {query!r}: {len(hits)} hits")
        best = []
        for title, url, snippet in hits:
            if left() < 1.0 or not alive():
                break
            p = checker.yes(f"{title}. {snippet}", criterion)
            decisions += 1
            best.append((p, title, url, snippet))
            if p >= sure:
                break
        best.sort(key=lambda b: -b[0])
        found += [dict(title=t, url=u, text=s, p=round(p, 3)) for p, t, u, s in best if p >= 0.5]
        if best and best[0][0] >= sure:
            status = "answered"
            break
        reads = [b for b in best if b[2] not in tried][:PAGES]
        tried.update(b[2] for b in reads)
        timeout = max(1.0, min(4.0, left() - 2.0))
        pages = list(pool.map(lambda b: text_of(page_text(b[2], timeout)), reads)) if reads and left() > 2.5 else []
        top = 0.0
        for (_, title, url, _), text in zip(reads, pages):
            for passage in passages(text, keys, PASSAGES_PER_PAGE):
                if left() < 0.8 or not alive():
                    break
                p = checker.yes(f"{title}: {passage}", criterion)
                decisions += 1
                top = max(top, p)
                if p >= 0.5:
                    found.append(dict(title=title, url=url, text=passage, p=round(p, 3)))
                if p >= sure:
                    break
            if top >= sure:
                break
        if top >= min(sure, GOOD_ENOUGH):
            status = "answered"
            break
        if rnd == rounds or left() < 3.0:
            break
        try:
            query = gemma(
                "You are helping a voice assistant search the web. The question is:\n"
                f"{question}\n\nQueries tried so far: {queries}\nBest snippets found so far (they did "
                "not clearly answer it):\n" + "\n".join(f"- {s[:200]}" for s in ([b[3] for b in best] + [f["text"] for f in found])[:4])
                + "\n\nWrite one better, specific web search query about exactly what the question asks. "
                  "Keep the exact title or name from the question in double quotes; never switch to a "
                  "similarly named film, person or thing from the snippets (tools sim: Runner Runner drifted "
                  "to Runner (2026)). Reply with the query only.", 40).split("\n")[0][:150]
        except Exception as e:
            log("query refinement failed:", e)
            break
    found.sort(key=lambda f: -f["p"])
    return dict(status=status, kind=kind, evidence=found[:3], queries=queries, rounds=len(queries),
                decisions=decisions, ms=round((time.perf_counter() - t0) * 1000, 1))


def search_result_text(out):
    ev = out.get("evidence") or []
    if out["status"] == "answered":
        head = "Found. What the sources say:"
    elif ev:
        head = "Nothing confirmed it. Closest matches (say you couldn't confirm it):"
    else:
        return "Nothing reliable was found. Say you couldn't find it."
    lines = [head]
    for i, e in enumerate(ev[:3]):
        lines.append(f"{i + 1}. {e['title']} ({e['url']}): {e['text'].strip()[:900]}")
    return "\n".join(lines) + ("\nSay the source by site name if he asks; open a link with open_url only if he asks "
                               "to open it, and only a link from this list. Don't read links out.")


# ---- programs and web pages -------------------------------------------------------------------

class Apps:
    """The Start menu's apps (win32 and Store), launched the way the Start menu does."""

    def __init__(self):
        self.apps = []
        self.refresh()

    def refresh(self):
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
                                  "Get-StartApps | ConvertTo-Json -Compress"],
                                 capture_output=True, text=True, timeout=30, encoding="utf-8")
            rows = json.loads(out.stdout or "[]")
            self.apps = [(r["Name"], r["AppID"]) for r in (rows if isinstance(rows, list) else [rows])
                         if not re.search(r"uninstall|afinstaller|readme|help|support|website|manual", r["Name"], re.I)]
            log(f"apps: {len(self.apps)} in the Start menu")
        except Exception as e:
            log("app list failed:", e)

    # Windows here is Danish, so built-in apps have Danish Start menu names (Notesblok, Stifinder,
    # Lommeregner); each name below is tried in both languages (live run 8: "Notepad" wasn't found).
    ALIASES = {
        "vs code": ["visual studio code"], "vscode": ["visual studio code"], "code": ["visual studio code"],
        "chrome": ["google chrome"], "browser": ["google chrome"], "browseren": ["google chrome"],
        "notepad": ["notesblok", "notepad"], "notesblok": ["notesblok", "notepad"],
        "calculator": ["lommeregner", "calculator"], "lommeregner": ["lommeregner", "calculator"],
        "file explorer": ["stifinder", "file explorer"], "explorer": ["stifinder", "file explorer"],
        "stifinder": ["stifinder", "file explorer"], "finder": ["stifinder", "file explorer"],
        "settings": ["indstillinger", "settings"], "indstillinger": ["indstillinger", "settings"],
        "paint": ["paint"], "terminal": ["terminal"], "task manager": ["jobliste", "task manager"],
        "jobliste": ["jobliste", "task manager"], "snipping tool": ["klippeværktøj", "snipping tool"],
    }

    def find(self, name):
        n = name.lower().strip()
        names = [a[0].lower() for a in self.apps]
        for cand in self.ALIASES.get(n, []) + [n]:
            for i, a in enumerate(names):
                if a == cand:
                    return self.apps[i]
        for cand in self.ALIASES.get(n, []) + [n]:
            starts = [i for i, a in enumerate(names) if a.startswith(cand) or cand in a.split()]
            if starts:
                return self.apps[min(starts, key=lambda i: len(names[i]))]
        # The app id carries the English name even in a Danish Windows:
        # Microsoft.WindowsNotepad_..., Microsoft.WindowsCalculator_...
        key = re.sub(r"[^a-z0-9]", "", n)
        if len(key) >= 4:
            ids = [i for i, a in enumerate(self.apps) if key in re.sub(r"[^a-z0-9]", "", a[1].lower())]
            if ids:
                return self.apps[min(ids, key=lambda i: len(self.apps[i][1]))]
        close = difflib.get_close_matches(n, names, n=1, cutoff=0.75)
        return self.apps[names.index(close[0])] if close else None


def spotify_title():
    """'Artist - Song' while Spotify plays; None when it is paused or closed (the window is then
    titled "Spotify" / "Spotify Premium")."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "[Console]::OutputEncoding = [Text.Encoding]::UTF8; (Get-Process Spotify -ErrorAction "
                              "SilentlyContinue | Where-Object MainWindowTitle | Select-Object -First 1).MainWindowTitle"],
                             capture_output=True, text=True, timeout=10, encoding="utf-8").stdout.strip()
    except Exception:
        return None
    return out if out and not out.lower().startswith("spotify") else None


def launch_app(app_id, dry):
    if dry:
        return
    if app_id.startswith("http"):
        open_in_chrome(app_id, False)
    elif os.path.isabs(app_id) and os.path.exists(app_id):
        os.startfile(app_id)
    else:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])


def open_in_chrome(url, dry):
    if dry:
        return
    if CHROME.exists():
        subprocess.Popen([str(CHROME), url])
    else:
        os.startfile(url)


def resolve_url(target):
    t = target.strip()
    if re.match(r"(?i)^https?://", t):
        return t
    if re.match(r"(?i)^[\w-]+(\.[\w-]+)+(/\S*)?$", t):
        return "https://" + t
    hits, _ = search(t)
    key = re.sub(r"[^a-z0-9]", "", t.lower())
    for _, url, _ in hits[:8]:
        host = urllib.parse.urlsplit(url).netloc.lower().replace("www.", "")
        if key and key[:6] in re.sub(r"[^a-z0-9]", "", host):
            return f"https://{urllib.parse.urlsplit(url).netloc}/"
    return hits[0][1] if hits else "https://www.google.com/search?" + urllib.parse.urlencode({"q": t})


# ---- files ------------------------------------------------------------------------------------

def walk_roots():
    """(path, name, is_dir, mtime, haystack) for every file and folder under FILE_ROOTS. Runs in a
    child process (see Files)."""
    home = str(HOME).lower()
    entries = []
    for root in FILE_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith(".")]
            for name, is_dir in [(d, True) for d in dirnames] + [(f, False) for f in filenames]:
                p = os.path.join(dirpath, name)
                n = name.lower()
                entries.append((p, n, is_dir, _mtime(p), n + " " + p.lower().replace(home, "")))
            if len(entries) > MAX_FILES:
                return entries
    return entries


class Files:
    """Names and folders under Mikkel's user folders, rebuilt every 10 minutes. The walk runs in a
    child process: in this process it held the GIL for ~30 s each time and the checker's hint
    missed its 400 ms cap (grace-tools-sim v1)."""

    def __init__(self):
        self.entries = []  # (path, name_lower, is_dir, mtime, haystack)
        self.ready = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        while True:
            t0 = time.perf_counter()
            try:
                with ctx.Pool(1) as pool:
                    self.entries = pool.apply(walk_roots)
            except Exception as e:
                log("file index failed:", e)
            self.ready.set()
            log(f"files: {len(self.entries)} indexed in {time.perf_counter() - t0:.1f} s")
            time.sleep(600)

    def candidates(self, description, limit=20, kind="any"):
        self.ready.wait(60)
        q = description.lower()
        for k, v in SYNONYMS.items():
            if re.search(r"(?<![a-zæøå])" + re.escape(k) + r"(?![a-zæøå])", q):
                q += " " + v
        keys = [w for w in re.findall(r"[a-zæøå0-9]{2,}", q)
                if w not in {"den", "det", "der", "med", "fra", "som", "jeg", "min", "mit", "mine", "the", "and",
                             "file", "fil", "filen", "find", "lige", "sidste", "seneste", "nyeste", "har", "lavede", "gemt",
                             "en", "et", "og", "på", "af", "til", "om", "er", "jo", "så", "da", "of", "to", "my", "in", "on"}]
        exts = set().union(*[v for k, v in KIND_EXT.items() if k in q]) if any(k in q for k in KIND_EXT) else set()
        recent = bool(re.search(r"sidste|seneste|nyeste|i går|i dag|last|latest|recent", q))
        wants_dir = kind == "folder" or bool(re.search(r"mappe|folder|directory", q))
        now = time.time()
        scored = []
        for path, name, is_dir, mtime, hay in self.entries:
            hits = sum(1 for k in keys if k in hay or (len(k) > 5 and k[:5] in hay))
            if not hits:
                continue
            score = hits * 2 + sum(1 for k in keys if k in name)
            if wants_dir:
                score += 4 if is_dir else -4
            elif kind == "file" and is_dir:
                score -= 4
            if exts and not is_dir:
                score += 3 if os.path.splitext(name)[1] in exts else -2
            if recent:
                score += max(0.0, 2.0 - (now - mtime) / (14 * 86400))
            scored.append((score, path, is_dir, mtime))
        scored.sort(key=lambda s: -s[0])
        return scored[:limit]


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def explorer_navigate(path):
    """Shows `path` in the most recent File Explorer window. False if none is open."""
    ps = ("$w = @((New-Object -ComObject Shell.Application).Windows()) | Where-Object { $_.FullName -like '*explorer.exe' } "
          "| Select-Object -Last 1; if ($w) { $w.Navigate2($env:GRACE_PATH); $w.Visible = $true; 'ok' }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=10,
                             env={**os.environ, "GRACE_PATH": str(path)})
        return out.stdout.strip() == "ok"
    except Exception:
        return False


def inside_roots(path):
    try:
        rp = Path(path).resolve()
    except OSError:
        return False
    return any(rp == r.resolve() or r.resolve() in rp.parents for r in FILE_ROOTS if r.exists())


def describe(path, is_dir, mtime):
    when = datetime.fromtimestamp(mtime).strftime("%d %b %Y") if mtime else "?"
    rel = str(Path(path)).replace(str(HOME), "~")
    return f"{'folder' if is_dir else 'file'} {rel} (changed {when})"


# ---- the tools --------------------------------------------------------------------------------

DA_UNITS = {"nul": 0, "en": 1, "et": 1, "én": 1, "ét": 1, "to": 2, "tre": 3, "fire": 4, "fem": 5, "seks": 6,
            "syv": 7, "otte": 8, "ni": 9, "ti": 10, "elleve": 11, "tolv": 12, "tretten": 13, "fjorten": 14,
            "femten": 15, "seksten": 16, "sytten": 17, "atten": 18, "nitten": 19}
DA_TENS = {"tyve": 20, "tredive": 30, "fyrre": 40, "fyrretyve": 40, "halvtreds": 50, "halvtredsindstyve": 50,
           "tres": 60, "tresindstyve": 60, "halvfjerds": 70, "halvfjerdsindstyve": 70, "firs": 80,
           "firsindstyve": 80, "halvfems": 90, "halvfemsindstyve": 90}
DA_OPS = [("divideret med", "/"), ("delt med", "/"), ("ganget med", "*"), ("gange", "*"), ("plus", "+"),
          ("minus", "-"), ("i anden", "**2"), ("i tredje", "**3"), ("procent af", "/100*")]


DA_BIG = {"hundrede": 100, "hundred": 100, "tusind": 1000, "tusinde": 1000, "million": 1_000_000, "millioner": 1_000_000}


def da_word(w):
    """One Danish number word (as STT writes them): 'treogtyve' -> 23, 'tusind' -> 1000."""
    w = w.lower().replace("-", "")
    if w.isdigit():
        return int(w)
    if w in DA_BIG:
        return DA_BIG[w]
    if w in DA_UNITS:
        return DA_UNITS[w]
    if w in DA_TENS:
        return DA_TENS[w]
    u, sep, t = w.partition("og")
    if sep and u in DA_UNITS and t in DA_TENS:
        return DA_UNITS[u] + DA_TENS[t]
    for big, mult in DA_BIG.items():  # "tohundrede", "fem-tusind"
        if w.endswith(big) and w[: -len(big)] in DA_UNITS:
            return DA_UNITS[w[: -len(big)]] * mult
    return None


def from_danish(spoken):
    """'hvad er tolv tusind to hundrede og tretten gange to' -> '12213*2', or None if it isn't plain
    arithmetic. Numbers are built the usual way: units and tens add, hundrede multiplies the part
    before it, tusind/million close a group (tools sim: "tolv tusind to hundrede tretten" -> 12213)."""
    t = spoken.lower()
    # Danish writes 12.213 for twelve thousand two hundred and thirteen and 3,5 for three and a half
    # (live run 7: "hvad er 12.213" was read as 12.213).
    t = re.sub(r"(?<![\d.])(\d{1,3}(?:\.\d{3})+)(?![\d.,])", lambda m: m.group(1).replace(".", ""), t)
    t = " " + t.replace(",", ".") + " "
    for word, sym in DA_OPS:
        t = t.replace(f" {word} ", f" {sym} ")
    out, total, cur, open_num = [], 0, 0, False

    def flush():
        nonlocal total, cur, open_num
        if open_num:
            out.append(str(total + cur))
        total, cur, open_num = 0, 0, False

    for tok in t.split():
        if tok == "og" and open_num:
            continue
        n = None if re.fullmatch(r"\d+\.\d+", tok) else da_word(tok)
        if n is None:
            flush()
            if re.fullmatch(r"[0-9.]+|[-+*/()]|\*\*[23]|/100\*", tok):
                out.append(tok)
            continue
        if n >= 1000:
            total += (cur or 1) * n
            cur = 0
        elif n == 100:
            cur = (cur or 1) * 100
        else:
            if open_num and cur % 100 == 0 and cur and n < 100:
                cur += n
            elif open_num and (cur or total) and cur % 1000 == 0 and n < 1000:
                cur += n
            elif open_num and (cur or total):
                flush()
                cur = n
            else:
                cur += n
        open_num = True
    flush()
    expr = "".join(out)
    return expr if re.search(r"\d[-+*/]", expr) else None


WMO = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
       82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
       96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def weather(place, day):
    """Open-Meteo's forecast for a place: now, today, tomorrow or the day after."""
    g = json.loads(get("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": place, "count": 1, "language": "da"}), 8)).get("results")
    if not g:
        return None
    g = g[0]
    f = json.loads(get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": g["latitude"], "longitude": g["longitude"],
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
        "timezone": "Europe/Copenhagen", "forecast_days": 3, "wind_speed_unit": "ms"}), 8))
    where = f"{g['name']} ({g.get('country', '')})"
    if day == "now":
        c = f["current"]
        d = f["daily"]
        # Today's range too: "hvor varmt var der" went to a 10.8 s web search (live run 9).
        return (f"{where} right now: {WMO.get(c['weather_code'], 'weather code ' + str(c['weather_code']))}, "
                f"{c['temperature_2m']:.0f} degrees, wind {c['wind_speed_10m']:.0f} m/s. Today: "
                f"{d['temperature_2m_min'][0]:.0f} to {d['temperature_2m_max'][0]:.0f} degrees.")
    i = {"today": 0, "tomorrow": 1, "day_after": 2}.get(day, 0)
    d = f["daily"]
    return (f"{where}, {['today', 'tomorrow', 'the day after tomorrow'][i]} ({d['time'][i]}): "
            f"{WMO.get(d['weather_code'][i], 'weather code ' + str(d['weather_code'][i]))}, "
            f"{d['temperature_2m_min'][i]:.0f} to {d['temperature_2m_max'][i]:.0f} degrees, "
            f"{d['precipitation_probability_max'][i]}% chance of rain ({d['precipitation_sum'][i]:.1f} mm), "
            f"wind up to {d['wind_speed_10m_max'][i]:.0f} m/s.")


def safe_eval(expr):
    """Arithmetic only: numbers, + - * / // % ** and parentheses (no names, no calls)."""
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
           ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError("only numbers and + - * / % ** ( ) are allowed")

    return ev(ast.parse(expr.strip(), mode="eval").body)


def manifest():
    fn = lambda name, desc, props, req: {"type": "function", "function": {
        "name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}}
    s = lambda d, enum=None: {"type": "string", "description": d, **({"enum": enum} if enum else {})}
    return [
        fn("web_search", "Search the web and read the best pages (not for weather: use the weather tool). For anything "
           "current or local (news, opening hours, "
           "prices, who holds a job now), details you don't clearly know (a specific film, series, episode, person, "
           "product) and whenever he asks you to look something up. One search per question.",
           {"question": s("Mikkel's question as one full English sentence."),
            "query": s("Search query. English, except Danish for Danish local things (Danish places, shops, news)."),
            "type": s("fact: a specific fact. overview: what something is or is about (plots, episodes). "
                      "live: a current value (weather, opening hours, prices, scores, times). news: recent news.",
                      ["fact", "overview", "live", "news"]),
            "language": s("da for Danish local searches, otherwise en.", ["en", "da"])},
           ["query", "type"]),
        fn("weather", "Weather for a place: now, today, tomorrow or the day after, with temperature (min and max), "
           "rain and wind. Use it for every weather or temperature question, also 'how warm was it today'.",
           {"place": s("Town or city, e.g. Aarhus. If he doesn't say, use where he lives from your memory."),
            "day": s("When.", ["now", "today", "tomorrow", "day_after"])}, ["place", "day"]),
        fn("calculate", "Work out any arithmetic beyond the trivial exactly (never in your head). Pass his words in "
           "`spoken`; the tool reads Danish numbers itself.",
           {"spoken": s("Mikkel's words, exactly as he said them, e.g. 'hvad er sytten gange treogtyve'."),
            "expression": s("The same as digits, e.g. 17*23 (used only if the words can't be read).")}, ["spoken"]),
        fn("read_page", "Read a web page or article (a link from an earlier result) and return its text, to summarise "
           "or answer questions about it.", {"url": s("The page's link, from an earlier tool result.")}, ["url"]),
        fn("media", "Music and video on the PC. play: play a song or artist on Spotify now (query = song and/or "
           "artist). play_pause, next, previous, volume_up, volume_down, mute act on whatever is playing. "
           "now_playing says what Spotify is playing. search opens a Spotify search.",
           {"action": s("What to do.", ["play", "play_pause", "next", "previous", "volume_up", "volume_down", "mute",
                                         "now_playing", "search"]),
            "query": s("For play/search: the song and/or artist, e.g. 'Peace and Love Gilli'.")}, ["action"]),
        fn("open_app", "Open a program on Mikkel's PC by name (Spotify, Discord, VS Code, Word, Chrome...). "
           "Danish names work too: stifinder (File Explorer), lommeregner, indstillinger, notesblok.",
           {"name": s("The program's name.")}, ["name"]),
        fn("open_url", "Open a web page in a new tab in Mikkel's Chrome.",
           {"target": s("A full URL, a domain like wolt.com, or a site's name.")}, ["target"]),
        fn("find_files", "Find files or folders on Mikkel's PC (Desktop, Documents, Downloads, Pictures, Videos, Music). "
           "If he asked to open it, then open the clear best match with open_file without asking.",
           {"description": s("What he is looking for, in his words: name, topic, type, when."),
            "kind": s("folder if he asks for a folder (mappe), file if for a file, else any.", ["file", "folder", "any"])},
           ["description"]),
        fn("list_folder", "List what is in a folder: Desktop, Documents, Downloads, Pictures, or a path found earlier.",
           {"folder": s("Folder name (skrivebord/Desktop, Dokumenter, Overførsler/Downloads, Billeder) or a path.")}, ["folder"]),
        fn("open_file", "Open a file or folder found with find_files or list_folder, with its normal program.",
           {"path": s("The full path from find_files or list_folder.")}, ["path"]),
    ]


SLOW = ["web_search", "find_files", "read_page"]
BUILTIN_RISK = {"web_search": "read", "read_page": "read", "weather": "read", "calculate": "read",
                "find_files": "read", "list_folder": "read", "media": "visible", "open_app": "visible",
                "open_url": "visible", "open_file": "visible"}


def discovery_tools():
    s = registry.s
    return [
        {"type": "function", "function": {
            "name": "find_tools",
            "description": "You have many more tools than the ones listed: files (write, edit, move, compare, "
                           "convert), documents (Word, PDF, Excel, PowerPoint, charts), running code, git, "
                           "windows, system settings, network, processes, reminders and timers, watching things, "
                           "research, translation, downloads, transcription, the screen. Find the ones for a "
                           "task here, then run one with use_tool. Do this yourself; never ask him which tool.",
            "parameters": {"type": "object", "properties": {
                "need": s("What you need to do, in English, e.g. 'move files to a folder'.")}, "required": ["need"]}}},
        {"type": "function", "function": {
            "name": "use_tool",
            "description": "Run a tool found with find_tools.",
            "parameters": {"type": "object", "properties": {
                "name": s("The tool's name from find_tools."),
                "args": {"type": "object", "description": "Its arguments, as find_tools described them."}},
                "required": ["name", "args"]}}},
    ]


def full_manifest(all_tools=False):
    """The tools offered on every turn: the built-in ones, the core registry tools and the
    discovery pair (ADR 0021). all_tools offers every tool directly (for measuring the two)."""
    reg = [t.definition() for t in registry.TOOLS.values() if t.core or all_tools]
    return manifest() + reg + ([] if all_tools or len(registry.TOOLS) == len(reg) else discovery_tools())


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s", text, 1)[0]


class Suggester:
    """The registry tools that fit what he said (ADR 0021). On her own Gemma rarely looked past
    the fixed tool list (tools sim: 18/74 on files + code); ranking by embeddings found the
    right tool in the top 4 for 27/56 scenario lines, Gemma choosing from a one-line catalog
    for 54/56 at 129 ms p50. The catalog is a fixed prefix, so llama-server keeps it cached;
    the picks go into his message, never into the cached tool list."""

    def __init__(self):
        rows = [(n, first_sentence(f["function"]["description"])) for f in manifest() for n in [f["function"]["name"]]]
        rows += [(t.name, first_sentence(t.description)) for t in sorted(registry.TOOLS.values(), key=lambda t: (t.group, t.name))]
        self.offered = {f["function"]["name"] for f in manifest()} | {t.name for t in registry.TOOLS.values() if t.core}
        self.system = ("You route requests for a voice assistant on Mikkel's Windows PC. Tools:\n"
                       + "\n".join(f"{n}: {d}" for n, d in rows)
                       + "\n\nGiven what Mikkel said (Danish, transcribed), name up to 3 tools from the list that would "
                         "carry it out, best first, comma-separated. If none fits (chat, general knowledge), answer none.")

    def suggest(self, said, before=""):
        user = (f'Before that he said: "{before}"\n' if before else "") + f'Mikkel said: "{said}"'
        body = {"messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                "max_tokens": 24, "temperature": 0, "cache_prompt": True}
        req = urllib.request.Request(f"{LLM}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=5).read())["choices"][0]["message"]["content"]
        picks = [x.strip().strip("`.*") for x in out.split(",")][:3]
        # Tools she already has on every turn need no mention; only the hidden ones do.
        return [n for n in picks if n in registry.TOOLS and n not in self.offered]


def describe_tool(t):
    params = "; ".join(f"{k}: {v.get('description', '')}" + (f" (one of {', '.join(v['enum'])})" if v.get("enum") else "")
                       + ("" if k in t.required else " [optional]") for k, v in t.params.items())
    ask = " Needs his yes first (the tool asks)." if t.risk == "destructive" else ""
    return f"- {t.name}: {t.description}{ask} Args: {params or 'none'}"


class Tools:
    def __init__(self, checker, dry):
        self.checker, self.dry = checker, dry
        self.apps = Apps()
        self.files = Files()
        self.confirmations = registry.Confirmations()
        self.data_dir = ROOT / ".runtime/grace-data"
        self.scratch_dir = self.data_dir / "scratch"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.gemma = gemma

    def run(self, name, args, alive, confirmed=False):
        f = getattr(self, f"t_{name}", None)
        if f is not None:
            return f(args, alive)
        if name == "use_tool":
            inner = (args.get("name") or "").strip()
            if inner in ("use_tool", "find_tools"):
                return {"ok": False, "result": "use_tool runs tools found with find_tools."}
            if getattr(self, f"t_{inner}", None) is not None:
                return getattr(self, f"t_{inner}")(args.get("args") or {}, alive)
            name, args = inner, args.get("args") or {}
        try:
            out = registry.run(self, name, args, alive, confirmed)
        except Exception as e:
            log(f"tool {name} failed: {e!r}")
            return {"ok": False, "result": f"The {name} tool failed: {e}. Say it didn't work, or try another way."}
        if out is None:
            hits = registry.find(name.replace("_", " "), 3)
            alt = f" Maybe: {', '.join(t.name for t in hits)}." if hits else ""
            return {"ok": False, "result": f"There is no tool called {name}. Use find_tools.{alt}"}
        return out

    def confirm(self, token, alive):
        v = self.confirmations.take(token)
        if v is None:
            return {"ok": False, "result": "That confirmation has expired; nothing was done. Ask again if he still wants it."}
        name, args, summary, _ = v
        log(f"confirmed: {name} ({summary})")
        out = self.run(name, args, alive, confirmed=True)
        out["confirmed"] = summary
        return out

    def t_find_tools(self, a, alive):
        need = (a.get("need") or "").strip()
        hits = registry.find(need)
        if not hits:
            return {"ok": True, "result": "No tool matches that. Say you can't do it yet."}
        return {"ok": True, "tools": [t.name for t in hits],
                "result": "Tools you can run with use_tool:\n" + "\n".join(describe_tool(t) for t in hits)}

    def t_web_search(self, a, alive):
        query = (a.get("query") or a.get("question") or "").strip()
        question = (a.get("question") or query).strip()
        kind = a.get("type") if a.get("type") in ("fact", "overview", "news", "live") else "fact"
        language = a.get("language") if a.get("language") in ("en", "da") else "all"
        out = lookup(self.checker, question, query, kind, language, alive)
        log(f"web_search {kind} {out['status']} in {out['ms']:.0f} ms, {out['decisions']} decisions: {query!r}")
        return {"ok": True, "status": out["status"], "detail": out, "result": search_result_text(out)}

    def t_weather(self, a, alive):
        place = (a.get("place") or "Aarhus").strip()
        try:
            text = weather(place, a.get("day") or "today")
        except Exception as e:
            return {"ok": False, "result": f"The weather service didn't answer ({e}). Say so."}
        if not text:
            return {"ok": False, "result": f"No place called {place!r} was found. Say so."}
        return {"ok": True, "result": text}

    def t_calculate(self, a, alive):
        # Danish number words are read here, not by Gemma: she turned "treogtyve" into 30 (tools sim v3).
        # His transcribed words first (the engine adds them as `heard`): Gemma rewrote "12.213" as
        # "tolv tusinde to hundrede treogtyve" (tools sim run 7).
        expr = (from_danish(a.get("heard") or "") or from_danish(a.get("spoken") or "")
                or (a.get("expression") or "").replace(",", ".").replace("x", "*").replace("×", "*").replace("^", "**"))
        try:
            value = safe_eval(expr)
        except Exception as e:
            return {"ok": False, "result": f"Couldn't calculate {expr!r} ({e}). Say so."}
        text = f"{value:,.6g}" if isinstance(value, float) and not float(value).is_integer() else f"{int(value)}"
        return {"ok": True, "detail": {"expression": expr, "value": value},
                "result": f"{expr} = {text} (exact). If he disputes it, repeat this number and how you read his "
                          f"numbers (\"I read it as {expr}, which is {text}\"); don't agree with a different number."}

    def t_read_page(self, a, alive):
        url = (a.get("url") or "").strip()
        if not re.match(r"(?i)^https?://", url):
            return {"ok": False, "result": "That isn't a link. Use one from an earlier result."}
        page = page_text(url, 8)
        title = strip_tags((re.search(r"(?is)<title[^>]*>(.*?)</title>", page) or [None, ""])[1])
        text = text_of(page)
        if len(text) < 200:
            return {"ok": False, "result": "The page had no readable text. Say so."}
        return {"ok": True, "detail": {"url": url}, "result": f"{title}\n{text[:4000]}"}

    def t_media(self, a, alive):
        action = a.get("action") or "play_pause"
        q = re.sub(r"(?i)\b(by|af|med)\b", " ", (a.get("query") or "")).strip()
        q = " ".join(q.split())
        if action == "now_playing":
            title = spotify_title()
            return {"ok": True, "result": f"Spotify is playing: {title}." if title else "Spotify isn't playing anything."}
        if action == "play":
            # Live run 9: play_pause only resumed whatever was loaded, and Grace said she had started
            # the song he asked for. Opening a track link makes the Spotify app play that track.
            if not q:
                return {"ok": False, "result": "No song or artist was given. Ask what to play."}
            hits, _ = search(f"site:open.spotify.com/track {q}")
            tracks = [(t, u) for t, u, _ in hits if "open.spotify.com/track/" in u]
            if not tracks:
                return {"ok": False, "result": f"Couldn't find {q} on Spotify. Say so."}
            title, url = tracks[0]
            track_id = url.split("/track/")[1].split("?")[0].split("/")[0]
            if self.dry:
                return {"ok": True, "detail": {"track": url, "dry_run": True},
                        "result": f"Now playing: {title.split(' - song')[0]}."}
            os.startfile(f"spotify:track:{track_id}")
            for _ in range(10):
                time.sleep(0.5)
                playing = spotify_title()
                if playing:
                    break
            if not playing:
                return {"ok": False, "detail": {"track": url},
                        "result": "Opened the song in Spotify, but it isn't playing. Say so."}
            return {"ok": True, "detail": {"track": url}, "result": f"Now playing: {playing}."}
        if action == "search":
            if not q:
                return {"ok": False, "result": "Nothing to search for. Ask what to play."}
            if not self.dry:
                # Spaces as they are: the Spotify app showed "%20"-encoded words as "plus" (live run 9).
                os.startfile("spotify:search:" + q)
            return {"ok": True, "detail": {"dry_run": self.dry}, "result": f"Opened a Spotify search for {q}."}
        keys = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD}
        if action not in keys:
            return {"ok": False, "result": f"Unknown media action {action}."}
        if not self.dry:
            import ctypes

            for _ in range(5 if action.startswith("volume") else 1):  # one press is only 2 %
                ctypes.windll.user32.keybd_event(keys[action], 0, 0, 0)
                ctypes.windll.user32.keybd_event(keys[action], 0, 2, 0)
        if action == "play_pause" and not self.dry:
            time.sleep(0.6)
            playing = spotify_title()
            return {"ok": True, "result": (f"Pressed play/pause; Spotify is now playing {playing}." if playing else
                                           "Pressed play/pause; Spotify is paused now. This doesn't choose a song.")}
        said = {"play_pause": "Pressed play/pause. This doesn't choose a song.", "next": "Skipped to the next track.",
                "previous": "Went back a track.", "volume_up": "Turned the volume up.",
                "volume_down": "Turned the volume down.", "mute": "Toggled mute."}[action]
        return {"ok": True, "detail": {"dry_run": self.dry}, "result": said}

    def t_open_app(self, a, alive):
        name = (a.get("name") or "").strip()
        app = self.apps.find(name)
        if not app:
            # "åbn youtube" came in as open_app (tools sim): no such program, so open the site, but
            # only when a site of that name exists (live run 8 opened onlinenotepad.org for
            # "Notepad"), and say that it was the website.
            key = re.sub(r"[^a-z0-9]", "", name.lower())
            url = resolve_url(name)
            host = urllib.parse.urlsplit(url).netloc.lower().replace("www.", "")
            if not key or not host.startswith(key[:6]):
                return {"ok": False, "result": f"No program called {name!r} is installed. Say so."}
            open_in_chrome(url, self.dry)
            return {"ok": True, "detail": {"url": url, "dry_run": self.dry, "fallback": "website"},
                    "result": f"There's no {name} program, so I opened {host} in a new Chrome tab."}
        launch_app(app[1], self.dry)
        return {"ok": True, "detail": {"app": app[0], "dry_run": self.dry}, "result": f"Opened {app[0]}."}

    def t_open_url(self, a, alive):
        target = (a.get("target") or a.get("url") or "").strip()
        if not target:
            return {"ok": False, "result": "No page was given."}
        url = resolve_url(target)
        if not re.match(r"(?i)^https?://", url):
            return {"ok": False, "result": "That isn't a web address I can open."}
        open_in_chrome(url, self.dry)
        host = urllib.parse.urlsplit(url).netloc
        return {"ok": True, "detail": {"url": url, "dry_run": self.dry}, "result": f"Opened {host} in a new Chrome tab."}

    def t_find_files(self, a, alive):
        desc = (a.get("description") or "").strip()
        kind = a.get("kind") if a.get("kind") in ("file", "folder") else "any"
        t0 = time.perf_counter()
        cands = self.files.candidates(desc, kind=kind)
        if kind != "any":
            cands = [c for c in cands if c[2] == (kind == "folder")] or cands
        if not cands:
            return {"ok": True, "status": "not_found", "result": "No matching files or folders. Say you couldn't find it."}
        ranked = []
        for score, path, is_dir, mtime in cands[:8]:
            if not alive():
                break
            wanted = {"folder": "a folder", "file": "a file"}.get(kind, "a file or folder")
            p = self.checker.yes(describe(path, is_dir, mtime),
                                 f"Is this the {wanted.split()[-1]} Mikkel means? He asked for {wanted}: {desc}")
            ranked.append((p, path, is_dir, mtime))
        ranked.sort(key=lambda r: -r[0])
        good = [r for r in ranked if r[0] >= 0.5][:3] or ranked[:2]
        lines = [("Found (best first):" if ranked and ranked[0][0] >= 0.5 else
                  "No clear match. Closest (say you aren't sure):")]
        lines += [f"{i + 1}. {describe(path, d, m)} | path: {path}" for i, (p, path, d, m) in enumerate(good)]
        log(f"find_files {desc!r}: {len(cands)} candidates, top p {ranked[0][0] if ranked else 0:.2f} in {time.perf_counter() - t0:.1f} s")
        return {"ok": True, "status": "found" if ranked and ranked[0][0] >= 0.5 else "unsure",
                "detail": {"top": [(round(p, 3), path) for p, path, _, _ in ranked[:3]]},
                "result": "\n".join(lines) + "\nIf he asked to open it and the first result clearly fits, call open_file with its "
                          "path now, without asking. Otherwise say how many you found and where, naming at most two; "
                          "never read out a path."}

    def t_list_folder(self, a, alive):
        f = (a.get("folder") or "").strip()
        path = FOLDER_ALIASES.get(f.lower().strip(" /\\"))
        if path is None:
            # "skrivebordet/projects", "skrivebord projects": a known folder, then names below it
            # (tools sim: the file index matched "Semesterprojekt og Projektbeskrivelse" instead).
            parts = [x for x in re.split(r"[/\\]+|\s+", f.lower()) if x]
            if len(parts) > 1 and parts[0] in FOLDER_ALIASES:
                p = FOLDER_ALIASES[parts[0]]
                for part in parts[1:]:
                    subs = {c.name.lower(): c for c in p.iterdir() if c.is_dir()} if p.is_dir() else {}
                    p = subs.get(part) or next((v for k, v in subs.items() if k.startswith(part)), None)
                    if p is None:
                        break
                path = p
        if path is None:
            p = Path(os.path.expandvars(f.replace("~", str(HOME))))
            if p.is_dir() and inside_roots(p):
                path = p
            else:
                hits = [c for c in self.files.candidates(f, 50) if c[2]]
                # A folder named exactly what he said beats one that only contains the words.
                exact = [c for c in hits if Path(c[1]).name.lower() == f.lower().split("/")[-1].strip()]
                path = Path((exact or hits)[0][1]) if hits else None
        if path is None or not path.is_dir():
            return {"ok": False, "result": f"No folder called {f!r} was found. Say so."}
        items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        dirs = [p.name for p in items if p.is_dir()]
        files = [p.name for p in items if p.is_file() and p.name.lower() != "desktop.ini"]
        # Live run 9: 34 folders were listed as 25 names, and Grace counted the names.
        shown_d, shown_f = dirs[:120], files[:60]
        text = (f"{path.name} has exactly {len(dirs)} folders and {len(files)} files (count these numbers, not the "
                f"names below).\nFolders{' (first ' + str(len(shown_d)) + ')' if len(shown_d) < len(dirs) else ''}: "
                f"{', '.join(shown_d) or 'none'}\nFiles{' (first ' + str(len(shown_f)) + ')' if len(shown_f) < len(files) else ''}: "
                f"{', '.join(shown_f) or 'none'}")
        return {"ok": True, "detail": {"path": str(path), "folders": len(dirs), "files": len(files)}, "result": text}

    def t_open_file(self, a, alive):
        path = (a.get("path") or "").strip().strip('"')
        p = Path(os.path.expandvars(path.replace("~", str(HOME))))
        if not p.exists():
            hits = self.files.candidates(path, 1)
            p = Path(hits[0][1]) if hits else p
        if not p.exists() or not inside_roots(p):
            return {"ok": False, "result": "That file wasn't found in his folders. Say so."}
        if p.suffix.lower() in NO_OPEN:
            return {"ok": False, "result": "That is a program or script, which I don't open from here. Say so."}
        if not self.dry:
            # A folder goes into the Explorer window that is already open (live run 9: every folder
            # opened a new window); otherwise a new one.
            if not (p.is_dir() and explorer_navigate(p)):
                os.startfile(str(p))
        return {"ok": True, "detail": {"path": str(p), "dry_run": self.dry}, "result": f"Opened {p.name}."}


# ---- the server -------------------------------------------------------------------------------

def peer_alive(sock):
    """False once the engine has closed the connection (it gave up on this request)."""
    def alive():
        try:
            r, _, _ = select.select([sock], [], [], 0)
            if r:
                return sock.recv(1, socket.MSG_PEEK) != b""
            return True
        except OSError:
            return False
    return alive


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            for line in self.rfile:
                self.serve(json.loads(line))
        except (ConnectionError, OSError):
            return  # the engine dropped the request (barge-in or the hint's time cap)

    def serve(self, req):
        t0 = time.perf_counter()
        srv = self.server
        try:
            kind = req["kind"]
            if kind == "manifest":
                out = {"tools": full_manifest(srv.all_tools),
                       "slow": SLOW + [t.name for t in registry.TOOLS.values() if t.slow],
                       "risk": {**BUILTIN_RISK, **{t.name: t.risk for t in registry.TOOLS.values()}}}
            elif kind == "suggest":  # tools for this transcript, as text for his message
                hits = srv.suggester.suggest(req["said"], req.get("before") or "")
                out = {"tools": hits, "text": "\n".join(describe_tool(registry.TOOLS[n]) for n in hits)}
            elif kind == "events":  # due reminders, timers, watchers (grace_tools.time, docs/tools/time.md)
                from grace_tools import time as gt
                acked = [i for i in req.get("ack") or [] if gt.ack_event(srv.tools, i)]
                out = {"acked": acked, "events": gt.pending_events(srv.tools)}
            elif kind == "confirm":  # the engine heard his yes to a destructive call
                out = srv.tools.confirm(req["token"], peer_alive(self.connection))
            elif kind == "decide":
                # The engine checks every partial transcript; once a newer one for the same key is
                # waiting, an older one is pointless (live run 7: 1 in 4 hints came too late).
                key = req.get("key")
                wanted = lambda: True
                if key:
                    with srv.keys_lock:
                        gen = srv.keys[key] = srv.keys.get(key, 0) + 1
                    wanted = lambda: srv.keys.get(key) == gen
                try:
                    out = {"answer": srv.checker.decide(req["state"], req["question"], wanted)}
                except Superseded:
                    out = {"error": "superseded"}
            elif kind == "tool":
                out = srv.tools.run(req["name"], req.get("args") or {}, peer_alive(self.connection))
            elif kind == "lookup":  # the old lookup_server API, kept for bench/lookup_bench.py
                out = lookup(srv.checker, req["question"], req.get("query") or req["question"],
                             req.get("type", "fact"), req.get("language", "all"), peer_alive(self.connection))
            else:
                out = {"error": f"unknown kind {kind}"}
        except Exception as e:
            log("error:", repr(e))
            out = {"error": repr(e)}
        out.setdefault("ms", round((time.perf_counter() - t0) * 1000, 1))
        out["id"] = req.get("id")
        self.wfile.write((json.dumps(out, ensure_ascii=False) + "\n").encode())
        self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report what open_* would do, don't do it")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--all-tools", action="store_true", help="offer every tool directly (no find_tools)")
    args = ap.parse_args()
    grace_tools.load()
    server = Server(("127.0.0.1", args.port), Handler)
    server.checker = Checker()
    server.keys, server.keys_lock = {}, threading.Lock()
    server.all_tools = args.all_tools
    server.suggester = Suggester()
    server.tools = Tools(server.checker, args.dry_run)
    log(f"tools server listening on 127.0.0.1:{args.port}{' (dry run)' if args.dry_run else ''}")
    server.serve_forever()


if __name__ == "__main__":
    main()
