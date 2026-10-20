"""Tallies how replies open (first word or two), to spot a reaction used as a template.

    python bench/reaction_openers.py <bench-results/chat/<label>/transcripts.jsonl>
"""

import collections
import json
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
c = collections.Counter()
for l in lines:
    if l.strip():
        w = re.findall(r"[a-z']+", json.loads(l)["reply"].lower())
        c[" ".join(w[:2]) if w[:2] == ["oh", "no"] else (w[0] if w else "")] += 1
print(dict(c.most_common(8)), "of", sum(c.values()))
