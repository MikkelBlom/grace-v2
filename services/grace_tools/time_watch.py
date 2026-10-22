"""Watcher checks for the time group (grace_tools/time.py): files, processes, web pages, GitHub
Actions runs and URLs.

Each kind has start(w, now) -> (state, already) and check(w, state, now) -> (fired, say, state).
`w` is the watcher row (kind, target, cond, arg, label); `already` is set when the thing has
already happened, so no watcher is needed ("the page already says that"). Checks raise on
failure; the scheduler counts errors and gives up politely.

The OS layer (snapshot, process_names, fetch_text, http_status, gh) is a set of small module
functions that tests replace with monkeypatch.
"""

import hashlib
import html
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Grace-watcher (personal assistant, polite polling)"
PARTIAL = (".crdownload", ".part", ".partial", ".download", ".opdownload", ".tmp", ".!ut", ".aria2")
MAX_PAGE = 2_000_000


# ---- the OS layer -----------------------------------------------------------------------------

def snapshot(path):
    """name -> [size, mtime] for a folder's own entries (not recursive: cheap every few seconds),
    or for the file itself; {} if it doesn't exist."""
    p = Path(path)
    if p.is_file():
        st = p.stat()
        return {p.name: [st.st_size, st.st_mtime]}
    if not p.is_dir():
        return {}
    out = {}
    for e in p.iterdir():
        try:
            st = e.stat()
        except OSError:
            continue  # vanished mid-scan (a temp file renamed when a download completes)
        out[e.name] = [st.st_size if e.is_file() else -1, st.st_mtime]
    return out


def process_names():
    """Running program names, lowercase without .exe."""
    import psutil
    names = set()
    for p in psutil.process_iter(["name"]):
        n = (p.info.get("name") or "").lower()
        names.add(n[:-4] if n.endswith(".exe") else n)
    return names


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "da,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read(MAX_PAGE)
        charset = r.headers.get_content_charset() or "utf-8"
    return page_text(raw.decode(charset, "replace"))


def http_status(url):
    """The HTTP status the URL answers with, or None if nothing answers."""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def gh(args, cwd=None):
    """Runs the GitHub CLI and returns its JSON output."""
    out = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=30, cwd=cwd,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout).strip()[:200] or f"gh exited {out.returncode}")
    return json.loads(out.stdout or "null")


# ---- helpers ----------------------------------------------------------------------------------

def page_text(page):
    """The visible text of an HTML page (scripts, styles and tags dropped)."""
    page = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", page)
    page = re.sub(r"(?s)<!--.*?-->", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h\d|tr|section|article)>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def digest(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def cap(text):
    return text[:1].upper() + text[1:]


def is_partial(name):
    return name.lower().endswith(PARTIAL)


def names(items, n=3):
    items = sorted(items)
    more = f" and {len(items) - n} more" if len(items) > n else ""
    return ", ".join(items[:n]) + more


def mb(size):
    return f"{size / 1e6:.1f} MB" if size >= 1e5 else f"{max(size, 0) // 1000} kB"


def contains(text, needle):
    return " ".join(needle.lower().split()) in " ".join(text.lower().split())


def new_lines(old, new, limit=200):
    """The first lines of the new text that weren't there before (what changed on a page)."""
    seen = set(old.splitlines())
    fresh = [l for l in new.splitlines() if l not in seen and len(l) > 3]
    return " / ".join(fresh)[:limit]


def is_local(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost") or host.endswith(".test")


# ---- files ------------------------------------------------------------------------------------
# cond: download_done (a new, finished file whose size held still across two checks, and no
# partial downloads left), new_file, changed, deleted. arg: part of the file name, optional.

def _match(name, arg):
    return not arg or arg.lower() in name.lower()


def start_file(w, now):
    base = snapshot(w["target"])
    if w["cond"] == "deleted":
        gone = [n for n in base if _match(n, w["arg"])] if w["arg"] else list(base)
        if not gone:
            return {"base": base}, f"{w['arg'] or w['label']} isn't there now, so it's already gone."
    return {"base": base, "last": None, "is_file": Path(w["target"]).is_file()}, None


def check_file(w, state, now):
    cur, base = snapshot(w["target"]), state["base"]
    cond, arg, label = w["cond"], w["arg"], w["label"]
    new = [n for n in cur if n not in base and not is_partial(n) and _match(n, arg)]
    if cond == "download_done":
        # Chrome writes "x.pdf.crdownload" and renames it when done; Firefox ".part"; Edge too.
        busy = [n for n in cur if is_partial(n) and (n not in base or cur[n] != base[n])]
        if new and not busy:
            sizes = {n: cur[n][0] for n in new}
            if state.get("last") == sizes:
                total = sum(max(s, 0) for s in sizes.values())
                what = f"{names(new)} ({mb(total)})" if len(new) == 1 else f"{len(new)} files: {names(new)}"
                return True, f"Your download is finished: {what}, in {label}.", state
            state["last"] = sizes
        else:
            state["last"] = None
        return False, None, state
    if cond == "new_file":
        if new:
            return True, f"A new file appeared in {label}: {names(new)}.", state
        return False, None, state
    if cond == "deleted":
        watched = [n for n in base if _match(n, arg)] if arg else list(base)
        gone = [n for n in watched if n not in cur]
        if gone:
            where = "" if state.get("is_file") else f" from {label}"
            return True, f"{names(gone)} {'was' if len(gone) == 1 else 'were'} deleted or moved{where}.", state
        return False, None, state
    added = [n for n in cur if n not in base and _match(n, arg)]
    removed = [n for n in base if n not in cur and _match(n, arg)]
    changed = [n for n in cur if n in base and cur[n] != base[n] and _match(n, arg)]
    if added or removed or changed:
        bits = [f"{len(v)} {k} ({names(v, 2)})" for k, v in (("new", added), ("changed", changed), ("removed", removed)) if v]
        return True, f"{cap(label)} changed: " + ", ".join(bits) + ".", state
    return False, None, state


# ---- processes --------------------------------------------------------------------------------

def _running(target):
    t = target.lower().removesuffix(".exe")
    return any(n == t or (len(t) >= 4 and n.startswith(t)) for n in process_names())


def start_process(w, now):
    up = _running(w["target"])
    if w["cond"] == "starts" and up:
        return {}, f"{w['label']} is already running."
    if w["cond"] == "stops" and not up:
        return {}, f"{w['label']} isn't running now."
    return {}, None


def check_process(w, state, now):
    up = _running(w["target"])
    if w["cond"] == "starts" and up:
        return True, f"{w['label']} has started.", state
    if w["cond"] == "stops" and not up:
        return True, f"{w['label']} has stopped (it's no longer running).", state
    return False, None, state


# ---- web pages --------------------------------------------------------------------------------
# cond: changes, contains (fires when the text appears), loses (fires when it disappears).

def start_web(w, now):
    try:
        text = fetch_text(w["target"])
    except Exception:
        return {"hash": None, "text": None}, None  # the first good check sets the baseline
    if w["cond"] == "contains" and contains(text, w["arg"]):
        return {}, f"The page already says \"{w['arg']}\"."
    if w["cond"] == "loses" and not contains(text, w["arg"]):
        return {}, f"The page doesn't say \"{w['arg']}\" now."
    return {"hash": digest(text), "text": text[:100_000]}, None


def check_web(w, state, now):
    text = fetch_text(w["target"])
    if w["cond"] == "contains":
        if contains(text, w["arg"]):
            return True, f"{cap(w['label'])} now says \"{w['arg']}\".", state
        return False, None, state
    if w["cond"] == "loses":
        if not contains(text, w["arg"]):
            return True, f"{cap(w['label'])} no longer says \"{w['arg']}\".", state
        return False, None, state
    if state.get("hash") is None:
        return False, None, {"hash": digest(text), "text": text[:100_000]}
    if digest(text) != state["hash"]:
        what = new_lines(state.get("text") or "", text)
        return True, f"{cap(w['label'])} has changed." + (f" New: {what}" if what else ""), state
    return False, None, state


# ---- GitHub Actions ---------------------------------------------------------------------------

FIELDS = "databaseId,status,conclusion,displayTitle,workflowName,headBranch,url"


def _repo_args(repo):
    if re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""):
        return ["-R", repo], None
    return [], repo or None  # a local checkout: gh finds the repo from its folder


def start_github(w, now):
    extra, cwd = _repo_args(w["target"])
    arg = (w["arg"] or "").strip()
    if arg.isdigit():
        run = gh(["run", "view", arg, "--json", FIELDS, *extra], cwd)
        runs = [run]
    else:
        runs = gh(["run", "list", "--limit", "10", "--json", FIELDS, *extra, *(["--branch", arg] if arg else [])], cwd) or []
    if not runs:
        return {}, "There are no GitHub Actions runs" + (f" on {arg}" if arg and not arg.isdigit() else "") + "."
    active = [r for r in runs if r.get("status") != "completed"]
    if not active:
        r = runs[0]
        return {}, f"The latest run, {_name(r)}, already finished: {r.get('conclusion') or 'unknown'}."
    r = active[0]
    return {"run": r["databaseId"], "name": _name(r)}, None


def _name(r):
    return f"{r.get('workflowName') or 'workflow'} \"{r.get('displayTitle') or ''}\" on {r.get('headBranch') or '?'}"


def check_github(w, state, now):
    extra, cwd = _repo_args(w["target"])
    r = gh(["run", "view", str(state["run"]), "--json", FIELDS, *extra], cwd)
    if r.get("status") == "completed":
        result = r.get("conclusion") or "unknown"
        tail = " It failed." if result == "failure" else ""
        return True, f"The GitHub Actions run {_name(r)} finished: {result}.{tail}", state
    return False, None, state


# ---- URLs coming up ---------------------------------------------------------------------------

def start_url(w, now):
    code = http_status(w["target"])
    if code is not None and code < 500:
        return {}, f"{w['target']} is already up (it answers with {code})."
    return {}, None


def check_url(w, state, now):
    code = http_status(w["target"])
    if code is not None and code < 500:
        return True, f"{cap(w['label'])} is up now (it answers with {code}).", state
    return False, None, state


STARTS = {"file": start_file, "process": start_process, "web": start_web, "github": start_github, "url": start_url}
CHECKS = {"file": check_file, "process": check_process, "web": check_web, "github": check_github, "url": check_url}
