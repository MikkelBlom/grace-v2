"""Scores echo removal in an aec_test run: how strongly each processed signal still contains the
speaker signal (GCC-PHAT peak) and how loud it is while the clip plays vs. in its pauses.

  python bench/aec_eval.py bench-results/aec/<run> [extra.wav ...]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

d = Path(sys.argv[1])
sr = 48000
lp = sf.read(d / "loopback.wav")[0]
# Experimental: alignment is estimated, not measured; trust the echo peak more than the levels.
names = ["raw.wav", "aec.wav"] + [a for a in sys.argv[2:]] + (["ours.wav"] if (d / "ours.wav").exists() else [])


def gcc(ref, sig, maxlag):
    n = 1 << int(np.ceil(np.log2(len(ref) + len(sig))))
    r = np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n))
    r /= np.abs(r) + 1e-12
    cc = np.fft.irfft(r, n)
    cc = np.concatenate([cc[-maxlag:], cc[: maxlag + 1]])
    k = int(np.argmax(np.abs(cc)))
    return (k - maxlag) / sr * 1000, abs(cc[k]) / np.std(cc)


seg = lp[int(1.6 * sr): int(12 * sr)]
loud = None
for name in dict.fromkeys(names):
    m = sf.read(d / name)[0]
    off, peak = gcc(seg, m[int(1.6 * sr): int(1.6 * sr) + len(seg) + 4 * sr], 4 * sr)
    # Levels while the (aligned) clip is loud vs. paused, 10 ms frames, medians.
    f = sr // 100
    shift = int(-off / 1000 * sr)
    n = min(len(lp) - max(shift, 0), len(m)) // f
    e_lp = np.array([np.mean(lp[i * f + shift: (i + 1) * f + shift] ** 2) if 0 <= i * f + shift < len(lp) - f else 0 for i in range(n)])
    e_m = np.array([np.mean(m[i * f:(i + 1) * f] ** 2) for i in range(n)])
    # Masks follow each file's own alignment.
    if True:
        ok = e_lp > 0
        loud = ok & (e_lp >= np.percentile(e_lp[ok], 70))
        pause = (e_lp > 0) & (e_lp <= np.percentile(e_lp[ok], 15))
    db = lambda x: 10 * np.log10(max(x, 1e-12))
    print(f"{name:<12} echo peak {peak:5.1f} sigma   clip loud {db(np.median(e_m[loud])):6.1f} dB   pauses {db(np.median(e_m[pause])):6.1f} dB")
