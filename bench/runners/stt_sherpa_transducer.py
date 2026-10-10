"""STT runner for sherpa-onnx transducer models (e.g. RyeAI/ekko-v1-tiny).

sherpa-onnx is the same runtime the Rust engine would use (the `sherpa-onnx` crate), so the
latency measured here is close to production.

  python stt_sherpa_transducer.py --repo RyeAI/ekko-v1-tiny --subdir onnx-sherpa --int8
"""

import argparse
from pathlib import Path

import sherpa_onnx
import soundfile as sf
from huggingface_hub import snapshot_download

from common import serve, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--subdir", default="onnx-sherpa")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--provider", default="cpu", help="cpu | cuda (needs a CUDA build of sherpa-onnx)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    def load():
        root = Path(snapshot_download(args.repo, allow_patterns=[f"{args.subdir}/*"])) / args.subdir
        suffix = ".int8.onnx" if args.int8 else ".onnx"
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(root / f"encoder{suffix}"),
            decoder=str(root / f"decoder{suffix}"),
            joiner=str(root / f"joiner{suffix}"),
            tokens=str(root / "tokens.txt"),
            num_threads=args.threads,
            provider=args.provider,
            model_type="nemo_transducer",
        )
        return recognizer, {
            "model": args.repo,
            "runtime": f"sherpa-onnx-{args.provider}{'-int8' if args.int8 else ''}",
            "threads": args.threads,
        }

    def handle(recognizer, req):
        audio, sr = sf.read(req["audio"], dtype="float32")
        elapsed = timer()
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        recognizer.decode_stream(stream)
        return {"text": stream.result.text.strip(), "ms": elapsed()}

    serve(load, handle)


if __name__ == "__main__":
    main()
