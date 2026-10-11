"""STT runner for Whisper models on CTranslate2 (faster-whisper): production-class latency.

Converts the Hugging Face checkpoint to CTranslate2 on first use (cached under
bench/.models/ct2/). VAD and timestamps are off: the engine hands Whisper one finished
utterance, so we measure exactly that.

  python stt_faster_whisper.py --model danish-foundation-models/edda-v0.2 --compute float16
"""

import argparse
import subprocess
import sys
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

from common import serve, timer

CACHE = Path(__file__).resolve().parent.parent / ".models" / "ct2"


def convert(model: str, quantization: str) -> Path:
    out = CACHE / f"{model.replace('/', '--')}-{quantization}"
    if not (out / "model.bin").exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "ctranslate2.converters.transformers", "--model", model,
             "--output_dir", str(out), "--quantization", quantization,
             "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"],
            check=True, stdout=sys.stderr,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--compute", default="float16", help="float16 | int8_float16 | int8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="da")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--prompt", default=None, help="initial_prompt: vocabulary / previous turn")
    ap.add_argument("--hotwords", default=None, help="faster-whisper hotwords")
    ap.add_argument("--no-fallback", action="store_true",
                    help="decode at temperature 0 only (faster-whisper otherwise retries hotter)")
    args = ap.parse_args()

    def load():
        path = convert(args.model, "float16" if args.compute == "float16" else "int8_float16")
        model = WhisperModel(str(path), device=args.device, compute_type=args.compute)
        return model, {"model": args.model, "runtime": f"ctranslate2-{args.compute}-{args.device}",
                       "beams": args.beams, "fallback": not args.no_fallback}

    def handle(model, req):
        audio, _ = sf.read(req["audio"], dtype="float32")
        elapsed = timer()
        extra = {"temperature": 0.0} if args.no_fallback else {}
        if args.prompt:
            extra["initial_prompt"] = args.prompt
        if args.hotwords:
            extra["hotwords"] = args.hotwords
        segments, _ = model.transcribe(
            audio, language=args.language, beam_size=args.beams, vad_filter=False,
            without_timestamps=True, condition_on_previous_text=False, **extra,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments)
        return {"text": text, "ms": elapsed(),
                "avg_logprob": min((s.avg_logprob for s in segments), default=0.0),
                "no_speech_prob": max((s.no_speech_prob for s in segments), default=1.0)}

    serve(load, handle)


if __name__ == "__main__":
    main()
