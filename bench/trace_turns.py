"""Per-turn summary of an engine trace: stage timings, scores, transcripts, interruptions.

  python bench/trace_turns.py traces/engine-<id>.jsonl
"""
import collections
import json
import sys

recs = []
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if r["target"].startswith("grace"):
        recs.append(r)
t0 = min(r["ts_us"] for r in recs)
turns = collections.defaultdict(list)
for r in recs:
    turns[r.get("turn") or "-"].append(r)
for k in sorted(turns, key=lambda k: min(r["ts_us"] for r in turns[k])):
    rs = sorted(turns[k], key=lambda r: r["ts_us"])
    out = []
    for r in rs:
        f = r.get("fields", {})
        st = f.get("stage") or r["name"]
        if r["kind"] == "span" and st != "turn":
            out.append(f"{st} {(r['ts_us'] - t0) / 1e6:.2f}s+{r['dur_us'] / 1e3:.0f}ms")
        elif r["kind"] == "event":
            val = f.get("probability", f.get("text", f.get("delay_ms", "")))
            out.append(f"[{r['name']} {(r['ts_us'] - t0) / 1e6:.2f}s {str(val)[:60]}]")
    print(f"{k}: " + " | ".join(out))
    print()
