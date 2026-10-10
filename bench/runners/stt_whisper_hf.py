"""STT runner for any Whisper-architecture model on Hugging Face (edda, Røst, Brage, stock).

Uses transformers on CUDA in fp16 with greedy decoding: good for comparing accuracy between
models. Latency here is transformers latency, not what whisper.cpp in the engine will do.

  python stt_whisper_hf.py --model danish-foundation-models/edda-v0.2
"""

import argparse

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import serve, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--language", default="da")
    ap.add_argument("--beams", type=int, default=1)
    args = ap.parse_args()

    def load():
        processor = WhisperProcessor.from_pretrained(args.model)
        model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
        return (processor, model), {"model": args.model, "runtime": "transformers-fp16-cuda", "beams": args.beams}

    def handle(state, req):
        processor, model = state
        audio, sr = sf.read(req["audio"], dtype="float32")
        elapsed = timer()
        feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.inference_mode():
            ids = model.generate(feats, language=args.language, task="transcribe", num_beams=args.beams, max_new_tokens=220)
        torch.cuda.synchronize()
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return {"text": text, "ms": elapsed()}

    serve(load, handle)


if __name__ == "__main__":
    main()
