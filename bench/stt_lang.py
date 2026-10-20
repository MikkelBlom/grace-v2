"""Danish-or-English input: which model tells the two apart, and which transcribes English?

Mikkel speaks to Grace in Danish or in English (ADR 0017); STT was forced to Danish. Two parts:

  lid  Language ID (restricted to da/en) per candidate Whisper, on whole clips and on their first
       1.5 s (live turns are often short), with latency. edda itself always says Danish
       (p_da = 1.0 on 60/60 English clips), so the ID needs another model.
  en   English WER and latency per candidate on FLEURS en (edda forced to English included).
       Also counts how often ekko (the Danish hallucination check) hears nothing in English
       speech, which would drop the turn.

  bench/.venv/Scripts/python bench/stt_lang.py lid
  bench/.venv/Scripts/python bench/stt_lang.py en

Sets: FLEURS da / en (`fetch_fixtures.py`, `--source fleurs-en`) and the first 60 CoRal v3
conversation test clips (spontaneous Danish: the hard case for language ID).
Results: bench-results/stt_lang_<part>.jsonl and a summary on stdout.
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
STT = ROOT / "fixtures" / "stt"
EDDA = ROOT / "bench" / ".models" / "ct2" / "danish-foundation-models--edda-v0.2-int8_float16"
KW = dict(beam_size=1, vad_filter=False, without_timestamps=True, condition_on_previous_text=False)
SHORT = 1.5  # seconds


def norm(t):
    t = t.lower().replace("-", " ")
    return re.sub(r"[^\w\s']", "", t).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], max(len(r), 1)


def clips(n_coral=60):
    for line in (STT / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if c["id"].startswith("fleurs"):
            lang = "en" if "en" in c["tags"] else "da"
            audio, _ = sf.read(STT / c["audio"], dtype="float32")
            yield "fleurs-" + lang, lang, c["text"], audio
    path = ROOT / "fixtures" / "coral-v3" / "conversation" / "test-00000-of-00006.parquet"
    for r in pq.read_table(path).slice(0, n_coral).to_pylist():
        data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly  # librosa's numba pin breaks on numpy 2.5

            data = resample_poly(data, 16000, sr).astype(np.float32)
        yield "coral-conv", "da", r["text"], data


def load(name):
    from faster_whisper import WhisperModel

    path = str(EDDA) if name == "edda" else name
    m = WhisperModel(path, device="cuda", compute_type="int8_float16")
    m.transcribe(np.zeros(16000, np.float32), language="da", **KW)
    return m


def p_en(model, audio):
    _, _, probs = model.detect_language(audio)
    p = dict(probs)
    da, en = p.get("da", 0.0), p.get("en", 0.0)
    return en / max(da + en, 1e-9)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def run_lid(models, out):
    data = list(clips())
    loaded = {m: load(m) for m in models}
    rows = []
    for set_name, lang, _, audio in data:
        row = dict(set=set_name, lang=lang, dur=len(audio) / 16000)
        for name, m in loaded.items():
            row[name], row[name + "_ms"] = timed(lambda: p_en(m, audio))
            row[name + "_short"] = p_en(m, audio[: int(SHORT * 16000)])
        rows.append(row)
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"errors at threshold p_en > 0.5 (whole clip / first {SHORT} s), latency p50")
    for name in models:
        line = [f"{name:16}"]
        for s in ("fleurs-da", "coral-conv", "fleurs-en"):
            rs = [r for r in rows if r["set"] == s]
            wrong = lambda k: sum((r[k] > 0.5) != (r["lang"] == "en") for r in rs)
            line.append(f"{s} {wrong(name):2}/{len(rs)} {wrong(name + '_short'):2}/{len(rs)}")
        line.append(f"{np.median([r[name + '_ms'] for r in rows]):5.0f} ms")
        print("  ".join(line))


def run_en(models, out):
    from speech_server import Check

    data = [c for c in clips(0) if c[0] == "fleurs-en"]
    check = Check()
    empty = sum(not any(ch.isalnum() for ch in check.transcribe(a)) for *_, a in data)
    print(f"ekko hears nothing on {empty}/{len(data)} English clips")
    rows = []
    for name in models:
        m = load(name)
        e = w = 0
        ms = []
        for _, _, ref, audio in data:
            segs, t = timed(lambda: list(m.transcribe(audio, language="en", **KW)[0]))
            hyp = " ".join(s.text.strip() for s in segs)
            de, dw = wer(ref, hyp)
            e, w = e + de, w + dw
            ms.append(t)
            rows.append(dict(model=name, ref=ref, hyp=hyp, ms=t))
        print(f"{name:40} WER {100 * e / w:5.1f} %   p50 {np.median(ms):4.0f} ms")
        del m
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def main():
    sys.path.insert(0, str(ROOT / "services"))
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["lid", "en"])
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    out = ROOT / "bench-results" / f"stt_lang_{args.part}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.part == "lid":
        run_lid(args.models or ["tiny", "base", "small", "large-v3-turbo"], out)
    else:
        run_en(args.models or ["edda", "large-v3-turbo", "large-v3"], out)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
