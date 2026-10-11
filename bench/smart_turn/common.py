"""Shared helpers for the Smart Turn Danish track: official preprocessing, parquet IO, metrics.

Preprocessing is copied from pipecat-ai/smart-turn train.py (BSD-2): keep the last 8 s, left-pad
with zeros, WhisperFeatureExtractor(chunk_length=8) with do_normalize=True.
"""
import io
import glob

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
N = 8 * SR

_fe = None


def feature_extractor():
    global _fe
    if _fe is None:
        from transformers import WhisperFeatureExtractor
        _fe = WhisperFeatureExtractor(chunk_length=8)
    return _fe


def last_8s(a):
    if len(a) > N:
        return a[-N:]
    if len(a) < N:
        return np.pad(a, (N - len(a), 0))
    return a


def features(a):
    """(80, 800) float32 features for one 16 kHz mono clip, exactly as the official script."""
    x = feature_extractor()(last_8s(a), sampling_rate=SR, return_tensors="np", padding="max_length",
                            max_length=N, truncation=True, do_normalize=True).input_features
    return x[0].astype(np.float32)


def decode(audio_cell):
    """Decode an HF Audio cell ({bytes, path}) to 16 kHz mono float32."""
    a, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        import soxr
        a = soxr.resample(a, sr, SR).astype(np.float32)
    return a


def parquet_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return files


def metrics(labels, probs, thr=0.5):
    """'complete' = positive class. false_turn_end = share of incomplete clips called complete."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(probs) > thr
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum()); tn = int((~pred & ~labels).sum())
    n = len(labels)
    return {
        "n": n,
        "acc": (tp + tn) / n if n else float("nan"),
        "prec_complete": tp / (tp + fp) if tp + fp else float("nan"),
        "rec_complete": tp / (tp + fn) if tp + fn else float("nan"),
        "false_turn_end": fp / (fp + tn) if fp + tn else float("nan"),
        "n_complete": tp + fn, "n_incomplete": fp + tn,
    }


class TorchFeatures:
    """Batched GPU/CPU re-implementation of the official preprocessing (WhisperFeatureExtractor with
    chunk_length=8, do_normalize=True on the left-padded 8 s window). Verified against features()."""

    def __init__(self, device="cuda"):
        import torch
        self.torch = torch
        fe = feature_extractor()
        self.device = device
        self.window = torch.hann_window(fe.n_fft, device=device)
        self.mel = torch.from_numpy(fe.mel_filters).float().to(device)  # (201, 80)
        self.n_fft, self.hop = fe.n_fft, fe.hop_length

    def __call__(self, x):
        """x: (B, 128000) float tensor of left-padded 8 s windows -> (B, 80, 800)."""
        torch = self.torch
        x = x.to(self.device, torch.float32)
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-7)
        st = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)
        mag = st[..., :-1].abs() ** 2
        mel = self.mel.T @ mag
        log = torch.clamp(mel, min=1e-10).log10()
        mx = log.amax(dim=(1, 2), keepdim=True)
        log = torch.maximum(log, mx - 8.0)
        return (log + 4.0) / 4.0
