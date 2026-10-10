"""TTS runners for the Danish voice bake-off.

  python tts.py piper --voice <path/to/da_DK-talesyntese-medium.onnx>
  python tts.py supertonic --style F1
  python tts.py chatterbox --repo CoRal-project/roest-v3-chatterbox-350m --variant turbo
  python tts.py chatterbox --repo CoRal-project/roest-v3-chatterbox-500m --variant multilingual

`ttfa_ms` is time to the first audio the backend can hand over. Non-streaming backends only
hand over the whole sentence, so for them ttfa == total. That is still the number that
matters: the engine streams the reply sentence by sentence, so the first sentence's
synthesis time is the user-facing delay.

The first request is a warm-up and is not special-cased: CUDA backends pay kernel
compilation on it, so expect a slow s01 and look at p50.
"""

import argparse
import wave

import numpy as np
import soundfile as sf

from common import serve, timer


def piper_backend(args):
    from piper import PiperVoice

    def load():
        voice = PiperVoice.load(args.voice)
        return voice, {"model": args.voice.replace("\\", "/").split("/")[-1], "runtime": "piper-onnx-cpu"}

    def handle(voice, req):
        elapsed = timer()
        first = None
        chunks, rate = [], 22050
        for chunk in voice.synthesize(req["text"]):
            if first is None:
                first = elapsed()
            rate = chunk.sample_rate
            chunks.append(chunk.audio_int16_bytes)
        total = elapsed()
        with wave.open(req["out"], "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"".join(chunks))
        return {"ttfa_ms": first or total, "total_ms": total}

    return load, handle


def supertonic_backend(args):
    from supertonic import TTS

    def load():
        tts = TTS(model="supertonic-3", intra_op_num_threads=args.threads)
        style = tts.get_voice_style(args.style)
        return (tts, style), {"model": f"supertonic-3/{args.style}/{args.lang}", "runtime": "onnx-cpu", "steps": args.steps}

    def handle(state, req):
        tts, style = state
        elapsed = timer()
        wav, _ = tts.synthesize(req["text"], voice_style=style, lang=args.lang, total_steps=args.steps)
        total = elapsed()
        tts.save_audio(wav, req["out"])
        return {"ttfa_ms": total, "total_ms": total}

    return load, handle


def chatterbox_backend(args):
    import torch
    from huggingface_hub import snapshot_download

    def load():
        model_dir = snapshot_download(
            args.repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )
        if args.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_local(model_dir, "cuda")
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_local(model_dir, device="cuda")
        return model, {"model": args.repo, "runtime": f"chatterbox-{args.variant}-torch-cuda"}

    def handle(model, req):
        elapsed = timer()
        with torch.inference_mode():
            if args.variant == "turbo":
                wav = model.generate(req["text"])
            else:
                wav = model.generate(req["text"], language_id=args.lang)
        torch.cuda.synchronize()
        total = elapsed()
        sf.write(req["out"], wav.squeeze().float().cpu().numpy().astype(np.float32), model.sr)
        return {"ttfa_ms": total, "total_ms": total}

    return load, handle


def kokoro_backend(args):
    """Kokoro-82M through sherpa-onnx: the runtime the Rust engine would use (`sherpa-onnx` crate).

    sherpa-onnx calls back per generated chunk, so ttfa is real first-chunk latency.
    Speaker ids follow the model's voice order (3 = af_heart, 16 = am_michael, 21 = bf_emma).
    """
    import sherpa_onnx
    from pathlib import Path

    def load():
        if args.provider == "cuda":
            # ONNX Runtime's CUDA provider needs cuBLAS/cuDNN DLLs on PATH; PyTorch ships matching ones.
            import os
            import torch

            os.environ["PATH"] = str(Path(torch.__file__).parent / "lib") + os.pathsep + os.environ["PATH"]
        root = Path(args.model_dir)
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(root / "model.onnx"),
                    voices=str(root / "voices.bin"),
                    tokens=str(root / "tokens.txt"),
                    data_dir=str(root / "espeak-ng-data"),
                    dict_dir=str(root / "dict"),
                    lexicon=str(root / "lexicon-us-en.txt"),
                ),
                num_threads=args.threads,
                provider=args.provider,
            ),
        )
        tts = sherpa_onnx.OfflineTts(config)
        return tts, {"model": f"kokoro-v1.0/sid{args.sid}", "runtime": f"sherpa-onnx-{args.provider}", "threads": args.threads}

    def handle(tts, req):
        elapsed = timer()
        first = []

        def on_chunk(samples, progress):
            if not first:
                first.append(elapsed())
            return 1  # keep generating

        audio = tts.generate(req["text"], sid=args.sid, speed=1.0, callback=on_chunk)
        total = elapsed()
        sf.write(req["out"], np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)
        return {"ttfa_ms": first[0] if first else total, "total_ms": total}

    return load, handle


def pocket_backend(args):
    """Kyutai Pocket TTS (CPU-first, streaming): ttfa is the first streamed chunk."""
    import torch

    def load():
        from pocket_tts import TTSModel

        model = TTSModel.load_model()
        state = model.get_state_for_audio_prompt(args.voice)
        return (model, state), {"model": f"pocket-tts/{args.voice}", "runtime": "torch-cpu"}

    def handle(st, req):
        model, state = st
        elapsed = timer()
        first, chunks = None, []
        for chunk in model.generate_audio_stream(state, req["text"]):
            if first is None:
                first = elapsed()
            chunks.append(chunk)
        total = elapsed()
        audio = torch.cat(chunks).float().numpy() if chunks else np.zeros(1, np.float32)
        sf.write(req["out"], audio, model.sample_rate)
        return {"ttfa_ms": first or total, "total_ms": total}

    return load, handle


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="backend", required=True)
    p = sub.add_parser("piper")
    p.add_argument("--voice", required=True)
    s = sub.add_parser("supertonic")
    s.add_argument("--style", default="F1")
    s.add_argument("--steps", type=int, default=8)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--lang", default="da")
    k = sub.add_parser("kokoro")
    k.add_argument("--model-dir", required=True)
    k.add_argument("--sid", type=int, default=3)
    k.add_argument("--threads", type=int, default=4)
    k.add_argument("--provider", default="cpu")
    pk = sub.add_parser("pocket")
    pk.add_argument("--voice", default="alba")
    c = sub.add_parser("chatterbox")
    c.add_argument("--repo", required=True)
    c.add_argument("--variant", choices=["turbo", "multilingual"], required=True)
    c.add_argument("--lang", default="da", help="language_id for the multilingual variant")
    args = ap.parse_args()

    backends = {
        "piper": piper_backend,
        "supertonic": supertonic_backend,
        "chatterbox": chatterbox_backend,
        "kokoro": kokoro_backend,
        "pocket": pocket_backend,
    }
    serve(*backends[args.backend](args))


if __name__ == "__main__":
    main()
