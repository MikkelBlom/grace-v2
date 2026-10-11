"""Danish TTS bake-off for CoRal Røst v3 Chatterbox (350m / 500m) on the GPU.

  python roest_bakeoff.py --model 350m --mode fast --label 350m-fast
  python roest_bakeoff.py --model 500m --mode stock --label 500m-stock
  python roest_bakeoff.py --model 500m --mode fast --s3gen meanflow --label 500m-fast-mf

Speaks 30 Danish sentences (fixtures/tts/sentences.da.txt + sentences.da.extra.txt) `--rounds`
times, sentence by sentence as Grace would, and records per sentence: time to first audio
(ttfa: first streamed chunk; stock mode is non-streaming so ttfa == total), total time, audio
length, RTF. Afterwards every WAV is transcribed with edda-v0.2 (faster-whisper fp16) and scored
by character error rate: the stability check (babbling, truncation, skipped words).

Results: bench-results/tts-da/<label>/{results.jsonl, summary.json, sNN_rR.wav}.
`--llm-load` keeps llama-server (127.0.0.1:8080) decoding during the run, as it would be while
Grace speaks her first sentence.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sentences():
    out = []
    for f in (ROOT / "fixtures/tts/sentences.da.txt", Path(__file__).with_name("sentences.da.extra.txt")):
        out += [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return out


def spell_numbers(t):
    """ASR writes numbers as digits; the references spell them out."""
    from num2words import num2words

    da = lambda n, **k: num2words(int(n), lang="da", **k)
    t = re.sub(r"(\d+),(\d+)", lambda m: f"{da(m[1])} komma {' '.join(da(d) for d in m[2])}", t)
    t = re.sub(r"den (\d+)\.?", lambda m: f"den {da(m[1], to='ordinal')}", t)
    return re.sub(r"\d+", lambda m: da(m[0]), t)


def norm(t):
    t = unicodedata.normalize("NFC", spell_numbers(t).lower())
    t = re.sub(r"[^\wæøå ]+", " ", t)
    return " ".join(t.split())


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    """Character error rate over letters only (word splits like "tre tusind" don't count)."""
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return edits(r, h) / max(1, len(r))


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    return edits(r, h) / max(1, len(r))


def llm_load(stop):
    import urllib.request

    body = json.dumps({"prompt": "Fortæl en lang historie om en kat i København.", "n_predict": 400,
                       "cache_prompt": False}).encode()
    while not stop.is_set():
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/completion", body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
        except Exception as exc:  # report, keep trying: the run is still valid without load
            print("llm-load:", exc, file=sys.stderr)
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["350m", "500m", "mtl-base"], required=True)
    ap.add_argument("--mode", choices=["stock", "fast", "fast-eager", "fast-offline"], default="fast",
                    help="stock = chatterbox generate(); fast = graphs + streaming; fast-eager = graphs for T3, eager S3Gen")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--s3gen", default="auto", help="auto | meanflow (500m: use the 2-step decoder) | <steps>")
    ap.add_argument("--first", type=int, default=16, help="tokens in the first streamed chunk (40 ms each)")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--voice", default=None, help="prompt WAV to clone (default: the repo's conds.pt)")
    ap.add_argument("--llm-load", action="store_true")
    ap.add_argument("--no-cfg", action="store_true", help="500m: skip classifier-free guidance (one row instead of two)")
    ap.add_argument("--prompt-tokens", type=int, default=None, help="S3Gen keeps only the last N prompt tokens")
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute CER/summary from an earlier run's transcripts")
    ap.add_argument("--asr-only", action="store_true", help="transcribe and score an earlier run's WAVs")
    args = ap.parse_args()
    if args.rescore or args.asr_only:
        return score(ROOT / "bench-results/tts-da" / args.label, transcribe=args.asr_only)

    out = ROOT / "bench-results/tts-da" / args.label
    out.mkdir(parents=True, exist_ok=True)
    texts = sentences()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    sync = lambda: (torch.cuda.synchronize(), time.perf_counter())[1]

    t0 = sync()
    if args.mode == "stock":
        from roest_fast import model_dir
        if args.model == "350m":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            tts = ChatterboxTurboTTS.from_local(model_dir("350m"), "cuda")
            gen = lambda t: tts.generate(t)
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            tts = ChatterboxMultilingualTTS.from_local(model_dir("500m"), device="cuda")
            gen = lambda t: tts.generate(t, language_id="da")
        if args.voice:
            tts.prepare_conditionals(args.voice)
        sr = tts.sr

        def speak(text):
            a = sync()
            wav = gen(text).squeeze().float().cpu().numpy()
            t = (sync() - a) * 1000
            return wav, t, t, None
    else:
        from roest_fast import Roest
        m = Roest(args.model, dtype=dtype, s3gen=args.s3gen, cfg=not args.no_cfg)
        if args.voice:
            m.set_voice(args.voice)
        sr = m.sr
        streamer = m.stream if args.mode == "fast" else m.stream_eager

        def speak_offline(text):
            a = sync()
            wav, _ = m.synthesize(text)
            t = (sync() - a) * 1000
            return wav, t, t, None

        def speak(text):
            a = sync()
            first, chunks, arrive = None, [], []
            kw = dict(first=args.first, chunk=args.chunk, prompt_tokens=args.prompt_tokens) if args.mode == "fast" else {}
            for c in streamer(text, **kw):
                arrive.append((time.perf_counter() - a) * 1000)  # chunk is already on the host
                if first is None:
                    first = arrive[0]
                chunks.append(c)
            total = (sync() - a) * 1000
            # playback starts at the first chunk; a later chunk that lands after the audio before
            # it has finished playing is an audible gap
            stall, play_end = 0.0, arrive[0]
            for t, c in zip(arrive, chunks):
                stall += max(0.0, t - play_end)
                play_end = max(play_end, t) + len(c) / sr * 1000
            speak.stall = stall
            return np.concatenate(chunks), first, total, len(chunks)

    load_ms = (sync() - t0) * 1000
    if args.mode == "fast-offline":
        speak = speak_offline  # noqa: F811 (whole utterance, no streaming: the synthetic-data path)
    import subprocess
    # Other work on the shared GPU (llama-server, STT, other tracks) skews timings: record it.
    gpu_busy = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    torch.cuda.reset_peak_memory_stats()
    a = sync()
    speak("Hej, jeg er klar.")  # warm-up: graph capture and CUDA kernel load land here
    warm_ms = (sync() - a) * 1000

    stop = threading.Event()
    if args.llm_load:
        threading.Thread(target=llm_load, args=(stop,), daemon=True).start()
        time.sleep(1.0)

    rows = []
    for r in range(args.rounds):
        for i, text in enumerate(texts):
            try:
                wav, ttfa, total, chunks = speak(text)
            except Exception as exc:
                rows.append(dict(id=f"s{i:02d}_r{r}", text=text, error=f"{type(exc).__name__}: {exc}"))
                print("ERROR", text, exc, file=sys.stderr)
                continue
            path = out / f"s{i:02d}_r{r}.wav"
            sf.write(path, wav, sr)
            dur = len(wav) / sr
            row = dict(id=f"s{i:02d}_r{r}", text=text, ttfa_ms=round(ttfa, 1), total_ms=round(total, 1),
                       audio_s=round(dur, 3), rtf=round(total / 1000 / max(dur, 1e-3), 3), chunks=chunks, wav=str(path),
                       stall_ms=round(getattr(speak, "stall", 0.0), 1))
            rows.append(row)
            print(f"[{args.label}] {row['id']} ttfa={ttfa:.0f} total={total:.0f} audio={dur:.2f}s", file=sys.stderr)
    stop.set()
    vram = dict(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))

    ok = [r for r in rows if "wav" in r]
    p = lambda xs, q: round(float(np.percentile(xs, q)), 1) if xs else None
    ttfa = [r["ttfa_ms"] for r in ok]
    short = [r["ttfa_ms"] for r in ok if len(r["text"]) <= 40]
    summ = dict(
        label=args.label, model=args.model, mode=args.mode, dtype=args.dtype, s3gen=args.s3gen,
        first=args.first, chunk=args.chunk, cfg=not args.no_cfg, prompt_tokens=args.prompt_tokens, llm_load=args.llm_load, voice=args.voice or "builtin conds.pt",
        n=len(ok), errors=len(rows) - len(ok), load_ms=round(load_ms), warmup_ms=round(warm_ms), gpu_before=gpu_busy,
        ttfa_p50=p(ttfa, 50), ttfa_p95=p(ttfa, 95), ttfa_max=p(ttfa, 100), ttfa_short_p50=p(short, 50),
        total_p50=p([r["total_ms"] for r in ok], 50), rtf_p50=round(float(np.median([r["rtf"] for r in ok])), 3),
        stall_ms_total=round(sum(r.get("stall_ms", 0) for r in ok), 1),
        sentences_with_stall=sum(r.get("stall_ms", 0) > 0 for r in ok),
        secs_per_char_p50=round(float(np.median([r["audio_s"] / len(r["text"]) for r in ok])), 4), **vram,
    )
    write(out, rows, summ)  # timing is safe on disk before the ASR pass
    if args.mode == "stock":
        del tts
    else:
        del m
    torch.cuda.empty_cache()
    if args.no_asr:
        print(json.dumps(summ, ensure_ascii=False))
    else:
        score(out)


def write(out, rows, summ):
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")


def score(out, transcribe=True):
    """Transcribe every WAV of a run with edda-v0.2 (or reuse stored transcripts) and add CER/WER."""
    import librosa

    rows = [json.loads(l) for l in open(out / "results.jsonl", encoding="utf-8")]
    if transcribe:
        from faster_whisper import WhisperModel

        asr = WhisperModel(str(ROOT / "bench/.models/ct2/danish-foundation-models--edda-v0.2-float16"),
                           device="cuda", compute_type="float16")
        for row in rows:
            if "wav" not in row:
                continue
            audio, sr = sf.read(row["wav"], dtype="float32")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            segs, _ = asr.transcribe(audio, language="da", beam_size=5, vad_filter=False, temperature=0.0,
                                     without_timestamps=True, condition_on_previous_text=False)
            row["asr"] = " ".join(s.text.strip() for s in segs)
    for r in rows:
        if "asr" in r:
            r.update(cer=round(cer(r["text"], r["asr"]), 4), wer=round(wer(r["text"], r["asr"]), 4))
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    ok = [r for r in rows if "cer" in r]
    cers = [r["cer"] for r in ok]
    summ.update(cer_mean=round(float(np.mean(cers)), 4), cer_p50=round(float(np.median(cers)), 4),
                wer_mean=round(float(np.mean([r["wer"] for r in ok])), 4),
                bad_over_20pct_cer=sum(c > 0.2 for c in cers))
    write(out, rows, summ)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
