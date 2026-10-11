"""Turn TTS output (bench/synth/tts_da.py manifest) into Smart Turn parquet shards.

Sentences come from gen_sentences.py (local Gemma); voices from tts_da.py (Chatterbox multilingual,
MIT, cloning CC0 NST speakers, or Røst's built-in voice). complete -> endpoint True; incomplete and
filler variants -> False. Clips with high ASR CER (tts_da --check) are dropped. Every 10th sentence
goes to the test split, so no sentence is in both.

  python bench/smart_turn/build_synth.py fixtures/corpora/smart-turn/da-synth/wav/manifest.jsonl
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import build_coral as bc  # noqa: E402

OUT = "fixtures/corpora/smart-turn/da-synth"


def speech_end(a):
    """Last frame above floor + 15 dB, + 50 ms (TTS has clean silence, so this is reliable)."""
    hop = 160
    n = len(a) // hop
    fr = a[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-6)
    loud = np.nonzero(db > max(db.max() - 45, np.percentile(db, 5) + 15))[0]
    return ((loud[-1] + 5) * hop / bc.SR) if len(loud) else len(a) / bc.SR


def main():
    rng = random.Random(11)
    rows = {"train": [], "test": []}
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        if r.get("rejected") or r.get("error") or "wav" not in r:
            continue
        sid, kind = r["id"].rsplit("_", 1)
        split = "test" if int(sid[1:]) % 10 == 0 else "train"
        a, sr = sf.read(r["wav"], dtype="float32")
        x = bc.finish(a, speech_end(a), rng, [], False)
        rows[split].append({"audio": {"bytes": bc.encode(x), "path": f"{r['id']}.flac"},
                            "id": f"synth_{r['id']}", "language": "dan", "endpoint_bool": kind == "complete",
                            "midfiller": False, "endfiller": kind == "filler", "synthetic": True,
                            "dataset": "grace-synth-da", "kind": kind, "speaker": r["voice"], "text": r["text"]})
    for split, rr in rows.items():
        if not rr:
            continue
        os.makedirs(f"{OUT}/{split}", exist_ok=True)
        df = pd.DataFrame(rr)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f"{OUT}/{split}/synth.parquet",
                       row_group_size=200)
        print(split, len(df), df.kind.value_counts().to_dict(), "voices", df.speaker.nunique())


if __name__ == "__main__":
    main()
