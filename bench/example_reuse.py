"""Counts how often Grace reuses the wording of her prompt's example replies.

    python bench/example_reuse.py <system.md> <bench-results/chat/<label>/transcripts.jsonl>

Example replies are the quoted text after "->" in the prompt. A reply "reuses" an example when it
contains one of the example's sentences of four or more words verbatim (case-insensitive), or
the whole example reply. Short reactions ("Nice.", "Oh no!") are what the persona asks for, so
they don't count. Also reports the share of replies sharing any word 4-gram with an example.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


def main() -> None:
    prompt = Path(sys.argv[1]).read_text(encoding="utf-8")
    examples = re.findall(r'->\s*"([^"]+)"', prompt)
    sents = {norm(x) for e in examples for x in re.split(r"(?<=[.!?])\s+", e) if len(norm(x).split()) >= 4}
    sents |= {norm(e) for e in examples}
    grams = set()
    for e in examples:
        w = norm(e).split()
        grams |= {" ".join(w[i:i + 4]) for i in range(len(w) - 3)}
    replies = [json.loads(l)["reply"] for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if l.strip()]
    verbatim, gram_hits = [], 0
    for r in replies:
        n = norm(r)
        hit = [s for s in sents if s and s in n]
        if hit:
            verbatim.append((r, hit[0]))
        w = n.split()
        gram_hits += any(" ".join(w[i:i + 4]) in grams for i in range(len(w) - 3))
    print(json.dumps({
        "examples": len(examples),
        "replies": len(replies),
        "verbatim_reuse": len(verbatim),
        "any_4gram_overlap": gram_hits,
        "cases": [f"{h} <- {r}" for r, h in verbatim],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
