"""Smart Turn reference check: Python (HF features + onnxruntime) accuracy on bench-results/turn,
and the largest difference from the Rust results in results.tsv."""
import glob
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperFeatureExtractor

fe = WhisperFeatureExtractor(chunk_length=8)
s = ort.InferenceSession(".runtime/models/smart-turn-v3.2-fp32.onnx")
rust = {}
for line in open("bench-results/turn/results.tsv"):
    f, p, _ = line.split("\t")
    rust[Path(f).name] = float(p)
n = 8 * 16000
diffs, ok, tot = [], 0, 0
for f in sorted(glob.glob("bench-results/turn/*.wav")):
    a, _ = sf.read(f, dtype="float32")
    a = a[-n:] if len(a) > n else np.pad(a, (n - len(a), 0))
    x = fe(a, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=n,
           truncation=True, do_normalize=True).input_features.astype(np.float32)
    p = float(s.run(None, {"input_features": x})[0][0][0])
    diffs.append(abs(p - rust[Path(f).name]))
    ok += (p >= 0.5) == f.endswith("_complete.wav")
    tot += 1
print(f"python reference accuracy {100 * ok / tot:.0f}% on {tot} clips; rust-vs-python max diff {max(diffs):.4f}")
