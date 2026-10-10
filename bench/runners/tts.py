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
        return (tts, style), {"model": f"supertonic-3/{args.style}", "runtime": "onnx-cpu", "steps": args.steps}

    def handle(state, req):
        tts, style = state
        elapsed = timer()
        wav, _ = tts.synthesize(req["text"], voice_style=style, lang="da", total_steps=args.steps)
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
                wav = model.generate(req["text"], language_id="da")
        torch.cuda.synchronize()
        total = elapsed()
        sf.write(req["out"], wav.squeeze().float().cpu().numpy().astype(np.float32), model.sr)
        return {"ttfa_ms": total, "total_ms": total}

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
    c = sub.add_parser("chatterbox")
    c.add_argument("--repo", required=True)
    c.add_argument("--variant", choices=["turbo", "multilingual"], required=True)
    args = ap.parse_args()

    backends = {"piper": piper_backend, "supertonic": supertonic_backend, "chatterbox": chatterbox_backend}
    serve(*backends[args.backend](args))


if __name__ == "__main__":
    main()
