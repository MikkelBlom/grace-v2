"""Embedder bake-off for Grace's memory (ADR 0007): Danish questions must find English facts.

  bench/.venv-embed/Scripts/python.exe bench/embed_recall.py [model ...]

Reads fixtures/memory/recall.jsonl (40 target facts + 20 distractors, 43 queries) and reports
recall@1, recall@3 and MRR per model, plus CPU latency per query embedding (the embedder runs
on the CPU so it never competes with the LLM for the GPU). Appends to bench-results/summary.jsonl.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
QUERY_INSTRUCT = "Instruct: Given a spoken question, retrieve the stored fact that answers it\nQuery: "

# model -> (query kwargs, document kwargs) for SentenceTransformer.encode
MODELS = {
    "microsoft/harrier-oss-v1-270m": ({"prompt": QUERY_INSTRUCT}, {}),
    "microsoft/harrier-oss-v1-0.6b": ({"prompt": QUERY_INSTRUCT}, {}),
    "google/embeddinggemma-2": ({"prompt_name": "query"}, {"prompt_name": "document"}),
    "intfloat/multilingual-e5-small": ({"prompt": "query: "}, {"prompt": "passage: "}),
}


def main():
    rows = [json.loads(l) for l in (ROOT / "fixtures/memory/recall.jsonl").read_text(encoding="utf-8").splitlines()]
    facts = [r for r in rows if r["type"] == "fact"]
    queries = [r for r in rows if r["type"] == "query"]
    ids = [f["id"] for f in facts]
    names = sys.argv[1:] or list(MODELS)
    torch.set_num_threads(4)
    for name in names:
        qkw, dkw = MODELS[name]
        started = time.perf_counter()
        model = SentenceTransformer(name, device="cpu", model_kwargs={"torch_dtype": torch.float32})
        load_s = time.perf_counter() - started
        if "prompt_name" in qkw and qkw["prompt_name"] not in model.prompts:
            print(f"{name}: available prompts {list(model.prompts)}")
        doc = model.encode([f["text"] for f in facts], normalize_embeddings=True, **dkw)
        lat, ranks = [], []
        model.encode(["warm up"], normalize_embeddings=True, **qkw)
        for q in queries:
            t = time.perf_counter()
            v = model.encode([q["text"]], normalize_embeddings=True, **qkw)[0]
            lat.append((time.perf_counter() - t) * 1000)
            order = [ids[i] for i in np.argsort(-doc @ v)]
            ranks.append(order.index(q["target"]) + 1)
        ranks = np.array(ranks)
        da = np.array([q["lang"] == "da" for q in queries])
        res = {
            "kind": "embed", "label": name, "dim": int(doc.shape[1]),
            "recall@1": float((ranks == 1).mean()), "recall@3": float((ranks <= 3).mean()),
            "mrr": float((1 / ranks).mean()), "recall@1_da": float((ranks[da] == 1).mean()),
            "query_ms_p50": float(np.percentile(lat, 50)), "load_s": round(load_s, 1),
            "misses": [q["text"] for q, r in zip(queries, ranks) if r > 3],
        }
        print(json.dumps(res, ensure_ascii=False))
        with open(ROOT / "bench-results/summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
