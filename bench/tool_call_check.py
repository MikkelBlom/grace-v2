"""Does Gemma call web_search when she should? Grace's real system prompt + the web_search tool,
on the lookup and tool-choice items of fixtures/decide/scenarios.yaml (web_search vs no tool;
file/program/reminder/memory items count as "shouldn't search").

  bench/.venv/Scripts/python bench/tool_call_check.py [--hint-from bench-results/decide/<label>.jsonl]

--hint-from adds "[This likely needs a web lookup.]" to the user message where that decision run's
needs-lookup / tool choice said web_search, to measure a checker-assisted trigger.
"""

import argparse
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and read the best pages. Returns what the sources say about the query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Short, specific English search query."}}, "required": ["query"]},
    },
}


def ask(system, user):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": [TOOL], "max_tokens": 60, "temperature": 0, "cache_prompt": True}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req, timeout=60).read())
    msg = out["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    q = json.loads(calls[0]["function"]["arguments"]).get("query") if calls else None
    return bool(calls), q, (msg.get("content") or "")[:80], (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint-from")
    ap.add_argument("--label", default="tools")
    args = ap.parse_args()
    base = (ROOT / "config/prompts/system.md").read_text(encoding="utf-8")
    now = datetime.datetime(2026, 10, 21, 14)
    system = f"{base}\n\nToday is {now:%A %-d %B %Y}, around 14:00 local time (Denmark).\n\n" if sys.platform != "win32" else f"{base}\n\nToday is Wednesday 21 October 2026, around 14:00 local time (Denmark).\n\n"
    hints = {}
    if args.hint_from:
        for line in open(args.hint_from, encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] == "lookup":
                hints[r["id"]] = r["p"] > 0.5
            elif r["kind"] == "tool" and r["variant"] == "da":
                hints[r["id"]] = r["got"] == "web_search"
    doc = yaml.safe_load((ROOT / "fixtures/decide/scenarios.yaml").read_text(encoding="utf-8"))
    rows = []
    for it in doc["items"]:
        if it.get("kind") not in ("lookup", "tool"):
            continue
        should = it["expect"] is True or it["expect"] == "web_search"
        user = it["state"]
        if hints.get(it["id"]):
            user += "\n\n[This likely needs a web lookup.]"
        called, q, text, ms = ask(system, user)
        rows.append(dict(id=it["id"], should=should, called=called, query=q, text=text, ms=ms))
        mark = "ok " if called == should else "BAD"
        print(f"{mark} {it['id']:4} should={should!s:5} called={called!s:5} q={q!r} {text!r}")
    tp = sum(r["called"] and r["should"] for r in rows)
    fn = sum(not r["called"] and r["should"] for r in rows)
    fp = sum(r["called"] and not r["should"] for r in rows)
    print(f"searched when it should: {tp}/{tp + fn}; searched when it shouldn't: {fp}/{len(rows) - tp - fn}")
    out = ROOT / "bench-results" / "tools" / f"{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
