"""Builds the Danish STT fixture set used by `grace-bench stt`.

Writes 16 kHz mono WAVs plus `fixtures/stt/manifest.jsonl` (one clip per line). Audio is not
committed (dataset licences); rerun this script to recreate it. Sampling is seeded, so the same
clips come back every time.

  bench/.venv/Scripts/python bench/fetch_fixtures.py                  # FLEURS da (public)
  bench/.venv/Scripts/python bench/fetch_fixtures.py --source coral   # CoRal v3 (needs HF access)

Sources:
  fleurs  google/fleurs da_dk test: read Wikipedia sentences. Clean, public, not conversational.
  coral   CoRal-project/coral-v3 test: conversation + read-aloud. The closest public match to
          how people actually talk to Grace. Gated: request access on Hugging Face first.
"""

import argparse
import io
import json
import random
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "stt"
RATE = 16_000


def decode(audio: dict) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=RATE)
    return data


def sample_rows(path: str, text_col: str, n: int, seed: int):
    table = pq.read_table(path)
    idx = list(range(table.num_rows))
    random.Random(seed).shuffle(idx)
    rows = table.take(idx[:n]).to_pylist()
    return [(r, r[text_col]) for r in rows]


def fleurs(n: int, seed: int):
    path = hf_hub_download(
        "google/fleurs", "da_dk/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"
    )
    # FLEURS ids identify the sentence, not the recording: several speakers share one id.
    for i, (row, text) in enumerate(sample_rows(path, "transcription", n, seed)):
        yield f"fleurs-{row['id']}-{i:03}", row["audio"], row.get("raw_transcription") or text, ["fleurs", "read"]


def coral(n: int, seed: int):
    for subset, tag in (("conversation", "conversation"), ("read_aloud", "read")):
        path = hf_hub_download("CoRal-project/coral-v3", f"{subset}/test-00000-of-00006.parquet", repo_type="dataset")
        cols = pq.ParquetFile(path).schema_arrow.names
        text_col = "text" if "text" in cols else next(c for c in cols if "text" in c or "transcription" in c)
        for i, (row, text) in enumerate(sample_rows(path, text_col, n, seed)):
            clip_id = row.get("id_recording") or row.get("id") or i
            yield f"coral-{subset}-{clip_id}", row["audio"], text, ["coral", tag]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["fleurs", "coral"], default="fleurs")
    ap.add_argument("--n", type=int, default=60, help="clips per subset")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    audio_dir = OUT / args.source
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.jsonl"
    existing = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                clip = json.loads(line)
                existing[clip["id"]] = clip

    source = fleurs if args.source == "fleurs" else coral
    added = 0
    for clip_id, audio, text, tags in source(args.n, args.seed):
        rel = f"{args.source}/{clip_id}.wav"
        sf.write(OUT / rel, decode(audio), RATE, subtype="PCM_16")
        existing[clip_id] = {"id": clip_id, "audio": rel, "text": text.strip(), "tags": tags}
        added += 1

    with manifest.open("w", encoding="utf-8") as f:
        for clip in existing.values():
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")
    print(f"{added} clips from {args.source}; manifest has {len(existing)} -> {manifest}")


if __name__ == "__main__":
    main()
