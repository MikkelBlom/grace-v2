"""Build the Danish voice bank for synthetic speech: ~10 s cloning prompts from NST Danish (CC0).

  python build_voice_bank.py --speakers 32

Source: NST Danish ASR Database (Nordisk Språkteknologi, ~2000), CC0 1.0, distributed by
Språkbanken (https://www.nb.no/sprakbanken/?p=21492) and mirrored as
alexandrainst/nst-da on Hugging Face (CC0, not gated). Only small byte ranges of the parquet
shards are read: one 100-row group per speaker.

Not used, on purpose:
- CoRal (ASR) audio: OpenRAIL-D forbids speech-synthesis use.
- Common Voice da: since Oct 2025 only via Mozilla Data Collective, which needs an account.

Writes fixtures/corpora/voices/nst/<voice_id>.wav (16 kHz mono) and
fixtures/corpora/voices/manifest.jsonl (one line per voice: source, licence, speaker metadata,
the sentences the prompt contains).
"""

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/corpora/voices"
REPO = "datasets/alexandrainst/nst-da"
SOURCE = {
    "dataset": "NST Danish ASR Database (16 kHz), reorganised",
    "publisher": "Nordisk Språkteknologi / Språkbanken (National Library of Norway)",
    "url": "https://www.nb.no/sprakbanken/?p=21492",
    "mirror": "https://huggingface.co/datasets/alexandrainst/nst-da",
    "licence": "CC0-1.0",
}


def trim(a, sr, thresh_db=-40, pad=0.08):
    """Cut leading/trailing silence (frame RMS below thresh relative to peak)."""
    hop = int(sr * 0.01)
    frames = np.lib.stride_tricks.sliding_window_view(a, hop * 2)[::hop]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    idx = np.nonzero(db > thresh_db)[0]
    if not idx.size:
        return a[:0]
    s = max(0, idx[0] * hop - int(pad * sr))
    e = min(len(a), idx[-1] * hop + 2 * hop + int(pad * sr))
    return a[s:e]


def snr_db(a, sr):
    hop = int(sr * 0.02)
    frames = np.lib.stride_tricks.sliding_window_view(a, hop)[::hop]
    e = np.sort((frames ** 2).mean(axis=1))
    noise = e[: max(1, len(e) // 10)].mean() + 1e-12
    speech = e[-max(1, len(e) // 3):].mean()
    return 10 * np.log10(speech / noise)


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=32, help="candidates to fetch (balanced by sex afterwards)")
    ap.add_argument("--keep", type=int, default=24)
    ap.add_argument("--target-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    fs = HfFileSystem()
    shards = sorted(p for p in fs.ls(f"{REPO}/data", detail=False) if p.endswith(".parquet"))
    random.Random(args.seed).shuffle(shards)
    (OUT / "nst").mkdir(parents=True, exist_ok=True)
    cands, seen = [], set()
    for shard in shards:
        if len(cands) >= args.speakers:
            break
        pf = pq.ParquetFile(fs.open(shard, block_size=2 ** 20, cache_type="none"))
        md = pf.metadata
        # first row group of each speaker (reads only the tiny speaker_id column chunks)
        starts = {}
        for rg in range(md.num_row_groups):
            ids = set(pf.read_row_group(rg, columns=["speaker_id"]).column(0).to_pylist())
            if len(ids) == 1:
                starts.setdefault(ids.pop(), rg)
        for spk, rg in starts.items():
            if spk in seen or len(cands) >= args.speakers:
                continue
            seen.add(spk)
            rows = pf.read_row_group(rg).to_pylist()
            rows = [r for r in rows if r["speaker_id"] == spk]
            # full sentences only (NST also has single words and numbers)
            sents = [r for r in rows if len(r["text"].split()) >= 7 and r["text"].rstrip()[-1:] in ".?!"]
            clips, texts, total = [], [], 0.0
            for r in sents:
                a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                a = trim(a, sr)
                if len(a) < sr * 1.5:
                    continue
                clips.append(a)
                texts.append(r["text"])
                total += len(a) / sr
                if total >= args.target_s:
                    break
            if total < 6.0:
                print(f"skip speaker {spk}: only {total:.1f} s of sentences")
                continue
            gap = np.zeros(int(0.25 * sr), np.float32)
            audio = np.concatenate([x for c in clips for x in (c, gap)][:-1])
            peak = np.abs(audio).max()
            audio = audio / peak * 0.9 if peak > 0 else audio
            r0 = rows[0]
            cands.append(dict(speaker_id=spk, sex=r0["sex"].capitalize(), age=r0["age"], dialect=r0["dialect"],
                              recorded=r0["recording_datetime"], shard=shard.split("/")[-1], row_group=rg,
                              texts=texts, audio=audio, sr=sr, snr_db=round(float(snr_db(audio, sr)), 1)))
            print(f"speaker {spk} {r0['sex']} {r0['age']} {r0['dialect']}: {len(audio)/sr:.1f} s, snr {cands[-1]['snr_db']} dB")

    # balance: best-SNR speakers, alternating sex
    by_sex = {}
    for c in sorted(cands, key=lambda c: -c["snr_db"]):
        by_sex.setdefault(c["sex"], []).append(c)
    keep = []
    while len(keep) < args.keep and any(by_sex.values()):
        for sex in sorted(by_sex):
            if by_sex[sex] and len(keep) < args.keep:
                keep.append(by_sex[sex].pop(0))
    keep.sort(key=lambda c: (c["sex"], c["age"]))
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for i, c in enumerate(keep):
            vid = f"nst-{c['sex'][0].lower()}{c['age']:02d}-{c['speaker_id']}"
            path = OUT / "nst" / f"{vid}.wav"
            sf.write(path, c["audio"], c["sr"])
            entry = dict(voice_id=vid, file=str(path.relative_to(OUT)).replace("\\", "/"), sr=c["sr"],
                         seconds=round(len(c["audio"]) / c["sr"], 2), **{k: c[k] for k in (
                             "speaker_id", "sex", "age", "dialect", "recorded", "snr_db", "texts", "shard", "row_group")},
                         source=SOURCE,
                         note="Prompt for zero-shot cloning in synthetic training/test data only. "
                              "The speaker gave no consent to voice synthesis: do not clone with Røst "
                              "(its licence bans synthesising a natural person's voice without consent), "
                              "never publish the synthetic audio as this person.")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"kept {len(keep)} voices -> {OUT / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
