"""The transcription worker for the research group's `transcribe` tool. It runs in its own process
with the speech venv (bench/.venv: faster-whisper + CUDA), not inside the tools server, so the
model only holds VRAM while a job runs and a stuck job can be killed. It loads edda the same way
services/speech_server.py does. The module registers no tools (the package loader imports it, so
nothing heavy is imported at the top).

  bench/.venv/Scripts/python.exe services/grace_tools/research_stt.py transcribe <16k.wav> <out.json> [--language da|en|auto]
  bench/.venv/Scripts/python.exe services/grace_tools/research_stt.py bench edda|turbo|large-v3   (docs/tools/research.md)

transcribe prints `progress <seconds done> <seconds total>` lines while it works and writes
{"language", "model", "segments": [[start, end, text]], "seconds", "ms"} to out.json.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Worktrees have no bench/.models; the main checkout does.
MODELS = next((p for p in (ROOT / "bench/.models/ct2", Path.home() / "Desktop/Projects/Grace/bench/.models/ct2")
               if p.exists()), ROOT / "bench/.models/ct2")
EDDA = MODELS / "danish-foundation-models--edda-v0.2-int8_float16"
# English and other languages: Whisper large-v3-turbo (docs/tools/research.md: edda, a Danish
# fine-tune, is worse on English).
TURBO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
CHUNK_S = 10  # FLEURS long-form WER, edda: 6 s 10.0 %, 10 s 9.8 %, 15 s 11.4 %, 30 s 20.5 %


def load(name):
    from faster_whisper import WhisperModel
    if name == "edda":
        return WhisperModel(str(EDDA), device="cuda", compute_type="int8_float16")
    if name == "large-v3":
        return WhisperModel("Systran/faster-whisper-large-v3", device="cuda", compute_type="int8_float16")
    return WhisperModel(TURBO, device="cuda", compute_type="int8_float16")


def read_wav(path):
    import numpy as np
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        a = resample_poly(a, 16000, sr).astype(np.float32)
    return a


def pieces(audio, max_s=None):
    """(start, end) sample ranges of speech, cut at pauses, each at most max_s long. Each piece is
    transcribed on its own, without timestamps: whole-file decoding with timestamps doubled edda's
    WER (docs/tools/research.md)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    max_s = max_s or CHUNK_S
    speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, max_speech_duration_s=max_s,
                                                     speech_pad_ms=200))
    out = []
    for sp in speech:
        if out and sp["end"] - out[-1][0] <= max_s * 16000:
            out[-1][1] = sp["end"]
        else:
            out.append([sp["start"], sp["end"]])
    return out


def run(model, audio, language, progress=None):
    """[[start s, end s, text]] for every speech piece."""
    out = []
    total = len(audio) / 16000
    for a, b in pieces(audio):
        segs, _ = model.transcribe(audio[a:b], language=language, beam_size=1, vad_filter=False,
                                   without_timestamps=True, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        if text:
            out.append([round(a / 16000, 2), round(b / 16000, 2), text])
        if progress:
            progress(b / 16000, total)
    return out, None


def detect(audio):
    """'da' or 'en' (or another code) from the first 30 s, with the multilingual turbo model."""
    model = load("turbo")
    lang, prob, _ = model.detect_language(audio[:16000 * 30])
    return lang, prob, model


def transcribe(wav, out_json, language="auto"):
    t0 = time.perf_counter()
    audio = read_wav(wav)
    model = None
    if language not in ("da", "en"):
        language, prob, turbo = detect(audio)
        print(f"language {language} {prob:.2f}", flush=True)
        model = None if language == "da" else turbo
    name = "edda" if language == "da" else "turbo"
    model = model or load(name)
    total = len(audio) / 16000
    segs, _ = run(model, audio, language, lambda done, tot: print(f"progress {done:.0f} {tot:.0f}", flush=True))
    Path(out_json).write_text(json.dumps({"language": language, "model": name, "segments": segs, "seconds": total,
                                          "ms": (time.perf_counter() - t0) * 1000}, ensure_ascii=False), encoding="utf-8")


# ---- the comparison (public FLEURS clips only, never Mikkel's voice) ---------------------------

def words(t):
    return re.sub(r"[^\w\s]", " ", t.lower()).split()


def errors(ref, hyp):
    import numpy as np
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return int(d[-1])


def bench(name):
    """One model per process: CTranslate2 crashes on exit after a second model was loaded."""
    import numpy as np
    stt = Path.home() / "Desktop/Projects/Grace/fixtures/stt"
    rows = [json.loads(line) for line in (stt / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    sets = {"da": [r for r in rows if r["id"].startswith("fleurs-") and "en" not in r["tags"]],
            "en": [r for r in rows if "en" in r["tags"]]}
    audio = {r["id"]: read_wav(stt / r["audio"]) for rs in sets.values() for r in rs}
    gap = np.zeros(8000, np.float32)
    model = load(name)
    run(model, np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32), "da")  # warm up CUDA kernels
    for lang, rs in sets.items():
        err = n = 0
        t0 = time.perf_counter()
        for r in rs:
            segs, _ = run(model, audio[r["id"]], lang)
            err += errors(words(r["text"]), words(" ".join(s[2] for s in segs)))
            n += len(words(r["text"]))
        clip_ms = (time.perf_counter() - t0) * 1000 / len(rs)
        # Long-form: all clips of the set joined into one recording (~10 min), as a lecture would be.
        joined = np.concatenate([x for r in rs for x in (audio[r["id"]], gap)])
        t0 = time.perf_counter()
        segs, _ = run(model, joined, lang)
        long_s = time.perf_counter() - t0
        long_err = errors(words(" ".join(r["text"] for r in rs)), words(" ".join(s[2] for s in segs)))
        print(f"{name:9} {lang}  clips WER {err / n:6.1%}  {clip_ms:5.0f} ms/clip   long {len(joined) / 16000 / 60:4.1f} min: "
              f"WER {long_err / n:6.1%} in {long_s:5.1f} s ({len(joined) / 16000 / long_s:4.0f}x real time)", flush=True)
        if name == "turbo":
            hits = sum(model.detect_language(audio[r["id"]][:16000 * 30])[0] == lang for r in rs)
            print(f"language detection (turbo) {lang}: {hits}/{len(rs)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["bench"]:
        bench(sys.argv[2] if len(sys.argv) > 2 else "edda")
    elif sys.argv[1:2] == ["transcribe"]:
        lang = sys.argv[sys.argv.index("--language") + 1] if "--language" in sys.argv else "auto"
        transcribe(sys.argv[2], sys.argv[3], lang)
    else:
        sys.exit(__doc__)
    sys.stdout.flush()
    os._exit(0)  # skip CTranslate2's CUDA teardown, which can crash (exit 127) after the work is done
