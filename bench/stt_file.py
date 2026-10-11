"""Transcribes whole WAV files with edda-v0.2 (CTranslate2, from the bench cache) and scores them
against a reference text: WER, plus which words were inserted (e.g. Grace's echo leaking in).

  python bench/stt_file.py <reference.txt> <file.wav>...
"""
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

MODEL = Path(__file__).resolve().parent / ".models" / "ct2" / "danish-foundation-models--edda-v0.2-int8_float16"


def words(t):
    return re.sub(r"[^\w\s]", " ", t.lower()).split()


def align(ref, hyp):
    """Levenshtein with backtrace -> (errors, inserted words)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = range(n + 1)
    d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ins = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
        elif j > 0 and d[i, j] == d[i, j - 1] + 1:
            ins.append(hyp[j - 1])
            j -= 1
        else:
            i -= 1
    return d[n, m], ins[::-1]


ref_text = " ".join(l for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l and not l.startswith("#"))
ref = words(ref_text)
model = WhisperModel(str(MODEL), device="cuda", compute_type="int8_float16")
for f in sys.argv[2:]:
    a, sr = sf.read(f, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = resample_poly(a, 16000, sr).astype(np.float32)
    segs, _ = model.transcribe(a, language="da", beam_size=1, vad_filter=False, condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segs)
    err, ins = align(ref, words(text))
    print(f"\n{Path(f).name}: WER {err / len(ref):.0%}  ({len(ins)} inserted words)\n  {text}")
