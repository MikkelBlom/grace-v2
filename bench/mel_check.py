"""Checks grace-vad's Rust Whisper features + Smart Turn output against the Python reference
(Hugging Face WhisperFeatureExtractor + onnxruntime), on the same audio.

  python bench/mel_check.py <file.wav> <rust_feats.bin> <rust_probability>
"""
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperFeatureExtractor

wav, feats_bin, rust_p = sys.argv[1], sys.argv[2], float(sys.argv[3])
audio, sr = sf.read(wav, dtype="float32")
assert sr == 16000
n = 8 * 16000
audio = audio[-n:] if len(audio) > n else np.pad(audio, (n - len(audio), 0))
fe = WhisperFeatureExtractor(chunk_length=8)
ref = fe(audio, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=n,
         truncation=True, do_normalize=True).input_features[0].astype(np.float32)
ours = np.fromfile(feats_bin, dtype="<f4").reshape(80, -1)
print(f"shape ref {ref.shape} ours {ours.shape}")
print(f"features: max abs diff {np.abs(ref - ours).max():.2e}, mean abs diff {np.abs(ref - ours).mean():.2e}")
s = ort.InferenceSession(".runtime/models/smart-turn-v3.2-cpu.onnx")
p = float(s.run(None, {"input_features": ref[None]})[0][0][0])
print(f"probability: python {p:.4f}  rust {rust_p:.4f}  diff {abs(p - rust_p):.4f}")
