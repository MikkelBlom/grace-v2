"""Our harsh Danish test (bench/make_turn_fixtures.py: 60 FLEURS clips, complete vs cut at a random
40-70 % point) for any Smart Turn ONNX, with the official numpy preprocessing.

  python bench/smart_turn/eval_harsh.py <model.onnx> [more.onnx...]
"""
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

files = sorted(glob.glob("bench-results/turn/*.wav"))
assert files, "run bench/make_turn_fixtures.py first"
x = np.stack([common.features(sf.read(f, dtype="float32")[0]) for f in files])
y = np.array([f.endswith("_complete.wav") for f in files])
for model in sys.argv[1:]:
    p = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(None, {"input_features": x})[0][:, 0]
    m = common.metrics(y, p)
    print(f"{os.path.basename(model)}: acc {m['acc']:.3f}  complete recognised {int(((p > .5) & y).sum())}/{y.sum()}"
          f"  cut recognised {int(((p <= .5) & ~y).sum())}/{(~y).sum()}  false turn end {m['false_turn_end']:.3f}")
