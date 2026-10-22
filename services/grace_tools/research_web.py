"""The web part of the research group: a minimal copy of tools_server's SearXNG search and page
reading (importing tools_server would load OpenVINO), plus source ranking that prefers official
and primary sources. Every network read goes through get(), which tests replace with recorded
responses (fixtures/tools/recorded/research/).
"""

import html
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SEARXNG = "http://127.0.0.1:8888"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36"
HONEST_UA = "Grace/2 (local voice assistant)"
BLOCKED = "possibly blocked by security software"

# Official and primary sources first: governments, agencies, universities, journals, standards,
# then encyclopedias; social media and content farms last.
PRIMARY = re.compile(r"(^|\.)(gov|gov\.[a-z]{2}|mil|edu|ac\.[a-z]{2}|int|europa\.eu|un\.org|who\.int|oecd\.org|"
                     r"nature\.com|science\.org|thelancet\.com|bmj\.com|nejm\.org|arxiv\.org|doi\.org|"
                     r"ncbi\.nlm\.nih\.gov|w3\.org|ietf\.org|iso\.org|ieee\.org|"
                     r"retsinformation\.dk|borger\.dk|sundhed\.dk|sst\.dk|ssi\.dk|dst\.dk|ft\.dk|"
                     r"regeringen\.dk|stm\.dk|um\.dk|skat\.dk|virk\.dk|politi\.dk|dmi\.dk|"
                     r"ku\.dk|au\.dk|sdu\.dk|aau\.dk|dtu\.dk|ruc\.dk|cbs\.dk|itu\.dk)$")
REFERENCE = re.compile(r"(^|\.)(wikipedia\.org|britannica\.com|lex\.dk|denstoredanske\.lex\.dk|snl\.no)$")
LOW = re.compile(r"(^|\.)(pinterest\.[a-z.]+|quora\.com|facebook\.com|instagram\.com|tiktok\.com|x\.com|"
                 r"twitter\.com|youtube\.com|linkedin\.com|medium\.com|answers\.com|scribd\.com)$")
SKIP_EXT = (".pdf", ".zip", ".mp4", ".mp3", ".jpg", ".png", ".exe")


def log(*a):
    print(time.strftime("%H:%M:%S"), "research:", *a, file=sys.stderr, flush=True)


def open_url(url, timeout, method="GET", headers=None):
    """urlopen with a browser User-Agent, retried once with an honest one on 403: some sites
    (w3.org) refuse a browser UA from a non-browser, others refuse anything else."""
    for ua in (UA, HONEST_UA):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": ua, "Accept-Language": "da,en;q=0.8",
                                                                  **(headers or {})})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 403 or ua == HONEST_UA:
                raise


def get(url, timeout, limit=2_000_000):
    with open_url(url, timeout) as r:
        raw = r.read(limit)
        charset = r.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def is_blocked(e):
    """An SSL connection cut mid-handshake: on this PC that is the antivirus refusing a flagged site.
    Retrying doesn't help, so the caller skips it and says why."""
    return "UNEXPECTED_EOF_WHILE_READING" in str(e) or (
        isinstance(e, ssl.SSLEOFError) or isinstance(getattr(e, "reason", None), ssl.SSLEOFError))


def fetch(url, timeout=8):
    """(page html, None) or ("", why it failed)."""
    try:
        return get(url, timeout), None
    except Exception as e:
        why = BLOCKED if is_blocked(e) else (f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__)
        log("fetch failed:", url, why, e)
        return "", why


def strip_tags(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def text_of(page):
    page = re.sub(r"(?is)<(script|style|noscript|svg|head|nav|footer|form)\b.*?</\1>", " ", page)
    page = re.sub(r"(?s)<!--.*?-->", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", page)
    page = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in page.split("\n"))
    return "\n".join(line for line in lines if len(line) > 30)


def title_of(page):
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    return strip_tags(m.group(1)) if m else ""


def search(query, language="all"):
    """[(title, url, snippet)] from SearXNG; [] when it is down (Grace runs without it)."""
    try:
        d = json.loads(get(f"{SEARXNG}/search?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "language": language}), 8))
    except Exception as e:
        log("searxng failed:", e)
        return []
    return [(r.get("title", ""), r["url"], r.get("content") or "") for r in d.get("results", []) if r.get("url")]


def domain(url):
    host = urllib.parse.urlsplit(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def site_name(url):
    """What Grace says for a source: 'Wikipedia', 'sundhed.dk'."""
    d = domain(url)
    if d.endswith("wikipedia.org"):
        return "Wikipedia"
    return d


def words(s):
    return set(re.findall(r"[a-zæøå0-9]{3,}", s.lower()))


def score(hit, i, query):
    """Higher is better: search rank, then official/primary sources, an official site of the thing
    asked about (python.org for a Python question), reference works; social media last."""
    title, url, snippet = hit
    d = domain(url)
    s = -0.3 * i
    if PRIMARY.search(d):
        s += 3
    elif REFERENCE.search(d):
        s += 1.5
    if LOW.search(d):
        s -= 3
    label = d.split(".")[-2] if d.count(".") >= 1 else d
    if any(len(w) >= 4 and w in label for w in words(query)):
        s += 2
    if re.match(r"(docs|developer|support|help)\.", d):
        s += 1
    if url.lower().split("?")[0].endswith(SKIP_EXT):
        s -= 10  # not readable as a page here
    return s


def rank(hits, query, seen_domains=()):
    """Hits best first, one per site, sites already used last."""
    order = sorted(range(len(hits)), key=lambda i: -score(hits[i], i, query))
    out, used = [], set()
    for i in order:
        d = domain(hits[i][1])
        if d in used:
            continue
        used.add(d)
        out.append(hits[i])
    return [h for h in out if domain(h[1]) not in seen_domains] + [h for h in out if domain(h[1]) in seen_domains]


def passages(text, keys, n, size=700):
    """The n paragraphs of a page that share the most words with the question."""
    paras, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > size and cur:
            paras.append(cur)
            cur = ""
        cur += line + " "
    if cur:
        paras.append(cur)
    ranked = sorted(paras, key=lambda p: -len(words(p) & keys))
    return [p.strip()[:900] for p in ranked[:n] if words(p) & keys]
