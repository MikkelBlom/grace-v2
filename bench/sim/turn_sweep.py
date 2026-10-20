"""Runs grace-turn-sim for named policy variants and prints one table.

  python bench/sim/turn_sweep.py baseline recovery "recovery --min-commit-ms 600" ...

Each argument is `<policy> [extra grace-turn-sim flags]`; results land in
bench-results/turn-sim/<slug>/. Model results are cached, so re-runs are fast. Needs the speech
server (services/speech_server.py) for spans not yet in the cache.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "release" / "grace-turn-sim.exe"
COLS = [("early_cut", "early"), ("merged_silent", "m.silent"), ("merged_heard", "m.heard"),
        ("split", "split"), ("split_heard", "split.h"), ("unanswered", "unans"),
        ("latency_p50", "p50"), ("latency_p90", "p90"), ("under_800ms", "<0.8s"),
        ("llm_requests_per_scene", "llm/sc")]


def run(variant):
    policy, *extra = variant.split()
    slug = re.sub(r"[^a-z0-9]+", "-", variant.lower()).strip("-")
    out = ROOT / "bench-results" / "turn-sim" / slug
    subprocess.run([str(EXE), "--scenes", str(ROOT / "fixtures/corpora/scenes/turns"),
                    "--out", str(out), "--policy", policy, *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return json.loads((out / "summary.json").read_text())


def main():
    print(f"{'variant':52} {'set':6} " + " ".join(f"{h:>8}" for _, h in COLS))
    for variant in sys.argv[1:]:
        s = run(variant)
        for kind in ("multi", "single"):
            row = s[kind]
            cells = " ".join(f"{row[k] if row[k] is not None else '-':>8}" for k, _ in COLS)
            print(f"{variant[:52]:52} {kind:6} {cells}", flush=True)


if __name__ == "__main__":
    main()
