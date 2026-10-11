"""Builds the Smart Turn test set in bench-results/turn/: each FLEURS clip once complete (trailing
silence trimmed to 250 ms) and once cut off at a random 40-70 % point (an unfinished sentence).

  python bench/make_turn_fixtures.py
  cargo run --release -p grace-vad --example turn_lab -- bench-results/turn/*.wav > bench-results/turn/results.tsv
  python bench/turn_check.py
"""
import glob
import os
import random

import numpy as np
import soundfile as sf

random.seed(3)
os.makedirs("bench-results/turn", exist_ok=True)
for f in sorted(glob.glob("fixtures/stt/fleurs/*.wav")):
    a, sr = sf.read(f, dtype="float32")
    b = os.path.basename(f)[:-4]
    loud = np.nonzero(np.abs(a) > 0.02)[0]
    a = a[: loud[-1] + int(0.25 * sr)] if len(loud) else a
    sf.write(f"bench-results/turn/{b}_complete.wav", a, sr, subtype="FLOAT")
    cut = a[: int(len(a) * random.uniform(0.4, 0.7))]
    sf.write(f"bench-results/turn/{b}_cut.wav", np.concatenate([cut, np.zeros(int(0.25 * sr), np.float32)]), sr, subtype="FLOAT")
