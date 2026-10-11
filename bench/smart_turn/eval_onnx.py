"""Evaluate a Smart Turn ONNX model on parquet sets in the smart-turn-data schema
(audio, endpoint_bool, language, ...), with the official preprocessing (batched torch port of
WhisperFeatureExtractor, max diff 5e-5 vs the official numpy path), ONNX on CPU.

  python bench/smart_turn/eval_onnx.py --model .runtime/models/smart-turn-v3.2-fp32.onnx \
      --set st-test=fixtures/corpora/smart-turn/v3.2-test/data/*.parquet --tag baseline

Writes per-clip probabilities to bench-results/smart_turn/<tag>_<set>.parquet and per-language
metrics to bench-results/smart_turn/<tag>_<set>.json, and prints a table.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

OUT = "bench-results/smart_turn"


def windows_for_batch(cells):
    return np.stack([common.last_8s(common.decode(c)) for c in cells]).astype(np.float32)


def run_set(sess, files, workers, keep_cols, tf):
    import torch
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for f in files:
            pf = pq.ParquetFile(f)
            cols = [c for c in ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic",
                                "dataset"] + keep_cols if c in pf.schema_arrow.names]
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=cols + ["audio"]).to_pandas()
                cells = list(t["audio"])
                chunks = [cells[i:i + 32] for i in range(0, len(cells), 32)]
                probs = []
                for w in pool.map(windows_for_batch, chunks):
                    x = tf(torch.from_numpy(w)).cpu().numpy()
                    probs.append(sess.run(None, {"input_features": x})[0][:, 0])
                t = t.drop(columns=["audio"])
                t["prob"] = np.concatenate(probs) if probs else []
                rows.append(t)
            print(f"  {os.path.basename(f)}: {sum(len(r) for r in rows)} clips", flush=True)
    return pd.concat(rows, ignore_index=True)


def report(df, name):
    out = {"all": common.metrics(df.endpoint_bool, df.prob)}
    if "language" in df:
        for lang, g in df.groupby("language"):
            out[str(lang)] = common.metrics(g.endpoint_bool, g.prob)
    print(f"\n{name}")
    print(f"{'lang':8} {'n':>6} {'acc':>6} {'prec':>6} {'rec':>6} {'falseEnd':>8}")
    for k, m in out.items():
        print(f"{k:8} {m['n']:6d} {m['acc']:6.3f} {m['prec_complete']:6.3f} {m['rec_complete']:6.3f} "
              f"{m['false_turn_end']:8.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", action="append", required=True, help="name=glob")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda", help="device for the (verified) torch feature extractor")
    ap.add_argument("--keep", default="", help="extra columns to keep, comma-separated")
    a = ap.parse_args()
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 6
    sess = ort.InferenceSession(a.model, sess_options=so, providers=["CPUExecutionProvider"])
    os.makedirs(OUT, exist_ok=True)
    keep = [c for c in a.keep.split(",") if c]
    tf = common.TorchFeatures(a.device)
    for spec in a.set:
        name, pattern = spec.split("=", 1)
        df = run_set(sess, common.parquet_files(pattern), a.workers, keep, tf)
        df.to_parquet(f"{OUT}/{a.tag}_{name}.parquet")
        res = report(df, f"{a.tag} / {name}")
        json.dump(res, open(f"{OUT}/{a.tag}_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
