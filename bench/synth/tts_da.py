"""Synthetic Danish speech with many voices, for training and test data (not Grace's live voice).

Python:
    from tts_da import synthesize, voices
    samples, rate = synthesize("Hvad er klokken?", "nst-f24-404")   # float32 mono, 16 kHz

CLI (run with bench/.venv-roest):
    python tts_da.py lines.jsonl out_dir [--check] [--engine auto]
    python tts_da.py --list-voices

`lines.jsonl` holds one {"id": ..., "text": ..., "voice": ...} per line ("voice" may be omitted or
"random" for a seeded random bank voice). Each line becomes out_dir/<id>.wav (16 kHz mono PCM16)
and a row in out_dir/manifest.jsonl (text, voice, engine, seconds, and with --check the edda-v0.2
transcript and CER; lines that stay above --max-cer after retries are marked "rejected").

Voices: fixtures/corpora/voices/manifest.jsonl (built by build_voice_bank.py): ~10 s prompts of
CC0 NST Danish speakers, plus "default" (the model repo's built-in conditioning).

Engines (licence-driven, see docs/MODELS.md "TTS Danish (Røst v3)"):
- "mtl-base": Resemble AI's multilingual Chatterbox (MIT, no use restrictions). Clones bank voices.
- "500m" / "350m": CoRal Røst v3. Their OpenRAIL licence forbids synthesising a natural person's
  voice without that person's prior informed consent, which the NST speakers never gave, so they
  only speak the "default" voice here.
"auto" picks mtl-base for bank voices and Røst 500m for "default".
Output is Perth-watermarked (imperceptible) unless --no-watermark, so synthetic clips stay
identifiable if they ever leak out of the training set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
VOICES = ROOT / "fixtures/corpora/voices"
RATE = 16000
ROEST_ENGINES = {"350m", "500m"}

_models: dict = {}
_conds: dict = {}


def voices() -> dict:
    """voice_id -> manifest entry (bank voices) plus "default"."""
    out = {"default": {"voice_id": "default", "file": None}}
    man = VOICES / "manifest.jsonl"
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                out[v["voice_id"]] = v
    return out


def engine_for(voice_id: str, engine: str = "auto") -> str:
    if engine == "auto":
        return "500m" if voice_id == "default" else "mtl-base"
    if engine in ROEST_ENGINES and voice_id != "default":
        raise ValueError(f"Røst ({engine}) may not clone '{voice_id}': its licence bans synthesising a natural "
                         "person's voice without their consent. Use engine mtl-base or auto.")
    return engine


def _model(engine: str):
    if engine not in _models:
        import torch
        from roest_fast import Roest, Sampling

        s = Sampling(0.8, 1000, 0.95, 0.0, 1.2) if engine == "350m" else Sampling(0.8, 0, 1.0, 0.05, 2.0, 0.5)
        # The 2-step "meanflow" S3Gen decoder (MIT, from chatterbox-turbo) halves the time per clip
        # against the 10-step one at the same ASR error rate in the bake-off.
        _models[engine] = Roest(engine, dtype=torch.bfloat16, sampling=s,
                                s3gen="auto" if engine == "350m" else "meanflow")
    return _models[engine]


def _use_voice(m, engine: str, voice_id: str):
    key = (engine, voice_id)
    if key not in _conds:
        v = voices().get(voice_id)
        if v is None:
            raise KeyError(f"unknown voice '{voice_id}' (see --list-voices)")
        m.set_voice(VOICES / v["file"] if v["file"] else None)
        _conds[key] = m.tts.conds
    m.tts.conds = _conds[key]
    m.voice = voice_id


def synthesize(text: str, voice_id: str = "default", engine: str = "auto", watermark: bool = True,
               seed: int | None = None):
    """-> (float32 samples in [-1, 1], 16000). One sentence at a time works best."""
    import librosa
    import torch

    eng = engine_for(voice_id, engine)
    m = _model(eng)
    _use_voice(m, eng, voice_id)
    if seed is not None:
        torch.manual_seed(seed)
    wav, _ = m.synthesize(text, watermark=watermark)
    wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=m.sr, target_sr=RATE)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav.astype(np.float32), RATE


_asr = None


def transcribe(samples):
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel

        _asr = WhisperModel(str(ROOT / "bench/.models/ct2/danish-foundation-models--edda-v0.2-float16"),
                            device="cuda", compute_type="float16")
    segs, _ = _asr.transcribe(samples, language="da", beam_size=5, vad_filter=False, temperature=0.0,
                              without_timestamps=True, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs)


def main():
    import soundfile as sf
    from roest_bakeoff import cer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--engine", default="auto", choices=["auto", "mtl-base", "500m", "350m"])
    ap.add_argument("--check", action="store_true", help="transcribe each clip with edda-v0.2 and retry bad ones")
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-watermark", action="store_true")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    bank = voices()
    if args.list_voices:
        for v in bank.values():
            print(v["voice_id"], v.get("sex", ""), v.get("age", ""), v.get("dialect", ""),
                  v.get("source", {}).get("licence", "model built-in"))
        return
    if not (args.lines and args.out):
        ap.error("lines.jsonl and out_dir are required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    bank_ids = [v for v in bank if v != "default"]
    todo = [json.loads(l) for l in Path(args.lines).read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(out / "manifest.jsonl", "a", encoding="utf-8") as man:
        for i, item in enumerate(todo):
            voice = item.get("voice") or "random"
            if voice == "random":
                voice = rng.choice(bank_ids)
            row = dict(id=item["id"], text=item["text"], voice=voice, engine=engine_for(voice, args.engine))
            try:
                for attempt in range(args.retries + 1 if args.check else 1):
                    wav, rate = synthesize(item["text"], voice, args.engine, not args.no_watermark,
                                           seed=args.seed + 1000 * attempt + i)
                    if args.check:
                        hyp = transcribe(wav)
                        row.update(asr=hyp, cer=round(cer(item["text"], hyp), 4), attempts=attempt + 1)
                        if row["cer"] <= args.max_cer:
                            break
                path = out / f"{item['id']}.wav"
                sf.write(path, wav, rate, subtype="PCM_16")
                row.update(wav=str(path), seconds=round(len(wav) / rate, 3), rate=rate,
                           watermarked=not args.no_watermark)
                if args.check and row["cer"] > args.max_cer:
                    row["rejected"] = True
            except Exception as exc:  # one bad line should not end a batch; it is recorded
                row["error"] = f"{type(exc).__name__}: {exc}"
            man.write(json.dumps(row, ensure_ascii=False) + "\n")
            man.flush()
            print(f"{i + 1}/{len(todo)} {row['id']} {voice} {row.get('seconds', '-')}s cer={row.get('cer', '-')}"
                  f"{' ERROR ' + row['error'] if 'error' in row else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
