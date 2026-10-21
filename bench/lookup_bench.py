"""The lookup loop on questions with known answers (fixtures/lookup/questions.yaml), against
taking the first search hit. Needs services/lookup_server.py (and SearXNG, llama-server) running.

  bench/.venv/Scripts/python bench/lookup_bench.py --label v1
"""

import argparse
import json
import re
import socket
import statistics as st
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def call(req):
    with socket.create_connection(("127.0.0.1", int(__import__("os").environ.get("TOOLS_PORT", "7813"))), timeout=60) as s:
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            buf += s.recv(65536)
    return json.loads(buf)


def first_hit(query):
    d = json.loads(urllib.request.urlopen("http://127.0.0.1:8888/search?" + urllib.parse.urlencode({"q": query, "format": "json"}), timeout=15).read())
    r = d["results"][0] if d["results"] else {}
    return f"{r.get('title', '')} {r.get('content', '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    items = yaml.safe_load((ROOT / "fixtures/lookup/questions.yaml").read_text(encoding="utf-8"))
    rows = []
    for it in items:
        out = call({"id": 1, "kind": "lookup", "question": it["q"], "query": it["query"], "type": it.get("type", "fact")})
        top = out["evidence"][0]["text"] + " " + out["evidence"][0]["title"] if out.get("evidence") else ""
        ok = bool(re.search(it["a"], top, re.I))
        first = bool(re.search(it["a"], first_hit(it["query"]), re.I))
        rows.append(dict(q=it["q"], ok=ok, first=first, status=out.get("status"), rounds=out.get("rounds"),
                         decisions=out.get("decisions"), ms=out.get("ms"), queries=out.get("queries"), top=top[:200]))
        print(f"{'ok ' if ok else 'BAD'} first={'ok ' if first else 'bad'} {out.get('status'):9} r{out.get('rounds')} "
              f"{out.get('ms', 0) / 1000:4.1f}s  {it['q']}  -> {top[:90]!r}")
        time.sleep(1)  # be gentle with the search engines
    n = len(rows)
    ms = sorted(r["ms"] for r in rows)
    print(f"\nloop right: {sum(r['ok'] for r in rows)}/{n}; first hit right: {sum(r['first'] for r in rows)}/{n}; "
          f"answered: {sum(r['status'] == 'answered' for r in rows)}/{n}; time p50 {st.median(ms) / 1000:.1f} s, "
          f"max {ms[-1] / 1000:.1f} s; rounds p50 {st.median(r['rounds'] for r in rows)}")
    out = ROOT / "bench-results" / "lookup" / f"{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
