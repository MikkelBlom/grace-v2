"""Prompt-cache and speed stats from a llama-server log (logs/llama-server.log).

  python bench/llama_log_stats.py [logs/llama-server.log] [--since HH:MM]

Per request: the slot, how many prompt tokens had to be (re)processed, prefill and decode speed.
A large prompt count on a long conversation is a prompt-cache miss (live run 8: up to 3,100
tokens, 1.5 s). Prints the distribution and the worst requests.
"""

import re
import sys
from collections import defaultdict

PROMPT = re.compile(r"id\s+(\d+) \| task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
EVAL = re.compile(r"id\s+(\d+) \| task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
STOP = re.compile(r"id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+)")
FULL = re.compile(r"forcing full prompt re-processing|erased invalidated context checkpoint|full prompt", re.I)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "logs/llama-server.log"
    tasks = defaultdict(dict)
    full = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        if m := PROMPT.search(line):
            tasks[m[2]].update(slot=int(m[1]), prompt_ms=float(m[3]), prompt_n=int(m[4]))
        elif m := EVAL.search(line):
            tasks[m[2]].update(eval_ms=float(m[3]), eval_n=int(m[4]))
        elif m := STOP.search(line):
            tasks[m[2]].update(n_ctx=int(m[3]))
        elif FULL.search(line):
            full += 1
    rows = [t for t in tasks.values() if "prompt_n" in t]
    if not rows:
        print("no requests in", path)
        return
    by_slot = defaultdict(int)
    for t in rows:
        by_slot[t["slot"]] += 1
    pn = [t["prompt_n"] for t in rows]
    pre = [t["prompt_n"] / t["prompt_ms"] * 1000 for t in rows if t["prompt_n"] >= 200]
    dec = [t["eval_n"] / t["eval_ms"] * 1000 for t in rows if t.get("eval_n", 0) >= 20]
    misses = [t for t in rows if t["prompt_n"] > 1000]
    print(f"requests {len(rows)}  per slot {dict(by_slot)}  full-reprocess log lines {full}")
    print(f"prompt tokens processed: p50 {pct(pn, .5)}  p90 {pct(pn, .9)}  max {max(pn)}")
    print(f"requests re-reading > 1000 tokens: {len(misses)} ({len(misses) / len(rows):.0%}), "
          f"{sum(t['prompt_ms'] for t in misses) / 1000:.1f} s spent")
    if pre:
        print(f"prefill tok/s (>=200 tokens): p50 {pct(pre, .5):.0f}  p90 {pct(pre, .9):.0f}")
    if dec:
        print(f"decode tok/s (>=20 tokens): p50 {pct(dec, .5):.1f}  p10 {pct(dec, .1):.1f}")
    for t in sorted(misses, key=lambda t: -t["prompt_n"])[:5]:
        print(f"  slot {t['slot']}: {t['prompt_n']} tokens in {t['prompt_ms']:.0f} ms (context {t.get('n_ctx', '?')})")


if __name__ == "__main__":
    main()
