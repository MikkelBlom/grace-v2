"""Grace lookup server: the decision checker (JevK5 4B on the Arc iGPU) and the web lookup loop
(ADR 0019, docs/MODELS.md "Typed-decision model").

  bench/.venv-openvino/Scripts/python.exe services/lookup_server.py

Localhost TCP 127.0.0.1:7813, newline-delimited JSON, one request per line:
  {"id": 1, "kind": "decide", "state": "...", "question": {"type": "noul", "instructions": "..."}}
  -> {"id": 1, "answer": {"type": "noul", "noul": 0.93, "confidence": 0.93, ...}, "ms": 231.0}
  {"id": 2, "kind": "lookup", "question": "Who played Sean Parker in The Social Network?",
   "query": "Sean Parker actor The Social Network"}
  -> {"id": 2, "status": "answered" | "not_found", "evidence": [{"title", "url", "text", "p"}],
      "queries": [...], "rounds": 2, "decisions": 9, "ms": 4120.0}

The lookup loop does not take the first hit: each round searches (SearXNG, then DuckDuckGo's
HTML page, then Wikipedia), has the checker judge whether each snippet answers the question,
reads the most promising pages when no snippet does, and asks Gemma for a better query before
the next round. It stops at an answer the checker is sure of, or at the round/time budget.
"""

import html
import json
import re
import socketserver
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openvino as ov
from jevk5.prompt import LETTERS, answer, decision_options, prompt_text
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / ".runtime/models/jevk5-ov/4b-int8-stateless"
DEVICE = "GPU.0"  # the Arc iGPU: keeps the RTX for Gemma (decode 50 -> 48.5 tok/s, vs 33 on the RTX)
TEMPERATURE = 1.22  # JevK5 v0.3 4B's calibration temperature (model card)
PORT = 7813
SEARXNG = "http://127.0.0.1:8888"
LLM = "http://127.0.0.1:8080"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Grace-local/0.3"

SURE = 0.85  # checker probability that a passage answers the question: stop and answer
MAX_ROUNDS = 3
BUDGET_S = 12.0
SNIPPETS = 6  # judged per round
PAGES = 3  # read per round when no snippet answers
PASSAGES_PER_PAGE = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


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

    def _read(self, state, instructions, texts):
        ids = self.tok(prompt_text(state, instructions, texts), add_special_tokens=False)["input_ids"]
        ids = ids[-6000:]  # long pages are trimmed by the caller; this is only a guard
        x = np.array([ids], dtype=np.int64)
        feed = {"input_ids": x, "attention_mask": np.ones_like(x)}
        if "position_ids" in self.names:
            feed["position_ids"] = np.arange(x.shape[1], dtype=np.int64)[None, :]
        with self.lock:
            row = self.req.infer(feed)[self.compiled.outputs[0]][0, -1].astype(np.float64)
        z = np.array([row[self.letter_ids[i]] for i in range(len(texts))]) / TEMPERATURE
        p = np.exp(z - z.max())
        return list(p / p.sum()), len(ids)

    def decide(self, state, question):
        options = decision_options(question)
        if len(options) > len(LETTERS):
            raise ValueError("more than 16 options is not supported here")
        probs, tokens = self._read(state, question["instructions"], [t for _, t in options])
        return answer(question, {k: v for (k, _), v in zip(options, probs)}, tokens)

    def answers(self, question, passage):
        q = {"type": "noul", "instructions": f"Does the evidence directly state the answer to this exact question: {question}"}
        return self.decide(passage, q)["noul"]


def get(url, timeout, limit=1_500_000):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en,da;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(limit)
        charset = r.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def text_of(page):
    page = re.sub(r"(?is)<(script|style|noscript|svg|head|nav|footer|form)\b.*?</\1>", " ", page)
    page = re.sub(r"(?s)<!--.*?-->", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", page)
    page = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in page.split("\n"))
    return "\n".join(line for line in lines if len(line) > 30)


def search(query):
    """[(title, url, snippet)] from SearXNG, else DuckDuckGo's HTML page, else Wikipedia."""
    try:
        d = json.loads(get(f"{SEARXNG}/search?" + urllib.parse.urlencode({"q": query, "format": "json"}), 8))
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
            hits.append((text_of(m.group(2)) or html.unescape(re.sub("<[^>]+>", "", m.group(2))), url,
                         html.unescape(re.sub("<[^>]+>", "", m.group(3)))))
        if hits:
            return hits, "duckduckgo"
    except Exception as e:
        log("duckduckgo failed:", e)
    try:
        d = json.loads(get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5}), 8))
        hits = [(r["title"], "https://en.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
                 html.unescape(re.sub("<[^>]+>", "", r["snippet"]))) for r in d["query"]["search"]]
        return hits, "wikipedia"
    except Exception as e:
        log("wikipedia failed:", e)
    return [], "none"


def words(s):
    return {w for w in re.findall(r"[a-zæøå0-9]{3,}", s.lower())}


def passages(text, keys, n):
    """The n windows of the page that share the most words with the question."""
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


def better_query(question, queries, seen):
    prompt = (
        "You are helping a voice assistant search the web. The question is:\n"
        f"{question}\n\nQueries tried so far: {queries}\n"
        "Best snippets found so far (they did not clearly answer it):\n"
        + "\n".join(f"- {s[:200]}" for s in seen[:4])
        + "\n\nWrite one better, specific English web search query. Use exact names you learned "
          "from the snippets. Reply with the query only."
    )
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0.3}
    req = urllib.request.Request(f"{LLM}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=10).read())
    return out["choices"][0]["message"]["content"].strip().strip('"').split("\n")[0][:150]


def lookup(checker, question, query):
    t0 = time.perf_counter()
    left = lambda: BUDGET_S - (time.perf_counter() - t0)
    queries, tried, found, decisions = [], set(), [], 0
    keys = words(question + " " + query)
    pool = ThreadPoolExecutor(PAGES)
    status = "not_found"
    for rnd in range(1, MAX_ROUNDS + 1):
        queries.append(query)
        hits, engine = search(query)
        hits = [h for h in hits if h[1] not in tried][:SNIPPETS]
        log(f"round {rnd} [{engine}] {query!r}: {len(hits)} hits")
        best = []
        for title, url, snippet in hits:
            if left() < 1.0:
                break
            p = checker.answers(question, f"{title}. {snippet}")
            decisions += 1
            best.append((p, title, url, snippet))
            if p >= SURE:
                break
        best.sort(key=lambda b: -b[0])
        found += [dict(title=t, url=u, text=s, p=round(p, 3)) for p, t, u, s in best if p >= 0.5]
        if best and best[0][0] >= SURE:
            status = "answered"
            break
        # No snippet is enough: read the most promising pages.
        reads = [b for b in best if b[2] not in tried][:PAGES]
        tried.update(b[2] for b in reads)
        timeout = max(1.0, min(4.0, left() - 2.0))
        pages = list(pool.map(lambda b: _page(b[2], timeout), reads)) if reads and left() > 2.5 else []
        done = False
        for (p0, title, url, _), text in zip(reads, pages):
            for passage in passages(text, keys, PASSAGES_PER_PAGE):
                if left() < 0.8:
                    break
                p = checker.answers(question, f"{title}: {passage}")
                decisions += 1
                if p >= 0.5:
                    found.append(dict(title=title, url=url, text=passage, p=round(p, 3)))
                if p >= SURE:
                    done = True
                    break
            if done:
                break
        if done:
            status = "answered"
            break
        if rnd == MAX_ROUNDS or left() < 3.0:
            break
        try:
            query = better_query(question, queries, [b[3] for b in best] + [f["text"] for f in found])
        except Exception as e:
            log("query refinement failed:", e)
            break
    found.sort(key=lambda f: -f["p"])
    return dict(status=status, evidence=found[:3], queries=queries, rounds=len(queries),
                decisions=decisions, ms=round((time.perf_counter() - t0) * 1000, 1))


def _page(url, timeout):
    try:
        return text_of(get(url, timeout))
    except Exception as e:
        log("fetch failed:", url, e)
        return ""


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            req = json.loads(line)
            t0 = time.perf_counter()
            try:
                if req["kind"] == "decide":
                    out = {"answer": self.server.checker.decide(req["state"], req["question"])}
                elif req["kind"] == "lookup":
                    out = lookup(self.server.checker, req["question"], req.get("query") or req["question"])
                    log(f"lookup {out['status']} in {out['ms']:.0f} ms, {out['decisions']} decisions: {req['question']!r}")
                else:
                    out = {"error": f"unknown kind {req['kind']}"}
            except Exception as e:
                log("error:", repr(e))
                out = {"error": repr(e)}
            out.setdefault("ms", round((time.perf_counter() - t0) * 1000, 1))
            out["id"] = req.get("id")
            try:
                self.wfile.write((json.dumps(out, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
            except OSError:
                return  # the engine gave up on this request (barge-in)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    server = Server(("127.0.0.1", PORT), Handler)
    server.checker = Checker()
    log(f"lookup server listening on 127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
