"""Tracks the speaker->mic echo delay over time in an aec_test run (3 s windows, GCC-PHAT),
using the recorded clock timelines. Shows Bluetooth delay drift within a run.

  python bench/echo_delay_track.py bench-results/aec/<run>
"""
import json
import sys

import numpy as np
import soundfile as sf

d = sys.argv[1]
sr = 48000
mic = sf.read(d + "/raw.wav")[0]
lp = sf.read(d + "/loopback.wav")[0]
t = json.load(open(d + "/timelines.json"))
clock = (t["mic"]["stamps"][0][1] - t["loopback"]["stamps"][0][1]) * 1e-7  # mic start - lp start, s


def gcc(ref, sig, maxlag):
    n = 1 << int(np.ceil(np.log2(len(ref) + len(sig))))
    r = np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n))
    r /= np.abs(r) + 1e-12
    cc = np.fft.irfft(r, n)
    cc = np.concatenate([cc[-maxlag:], cc[: maxlag + 1]])
    k = int(np.argmax(np.abs(cc)))
    return (k - maxlag) / sr, abs(cc[k]) / np.std(cc)


rows = []
for s in np.arange(2.0, 17.0, 2.0):
    k = int(s * sr)
    w = int(3 * sr)
    j = int((s + clock) * sr)  # loopback index captured at the same instant as mic index k
    pre = int(0.5 * sr)  # search up to 500 ms of delay
    if j - pre < 0 or j + w > len(lp) or k + w > len(mic):
        continue
    lag, peak = gcc(lp[j - pre: j + w], mic[k: k + w], sr // 2)
    rows.append(f"{s:4.0f}s {(lag + 0.5) * 1000:6.0f} ms ({peak:4.1f} sigma)")
print(d.split("/")[-1] + ":\n  " + "\n  ".join(rows))
