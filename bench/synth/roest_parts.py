"""Per-stage timing of the fast Røst path, to separate model cost from GPU contention.

  python roest_parts.py --model 350m
  python roest_parts.py --model 500m --s3gen meanflow

The GPU is shared with llama-server and other work, so end-to-end runs vary with their load.
Here each stage repeats many times; `min` approximates an idle GPU, `p50` the load at the time.
Predicted ttfa = prefill + (first - 1) decode steps + first S3Gen window.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roest_fast import Roest  # noqa: E402


def timeit(fn, n):
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        a = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - a) * 1000)
    return dict(min=round(min(out), 2), p50=round(float(np.median(out)), 2))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="350m")
    ap.add_argument("--s3gen", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--first", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=None)
    ap.add_argument("--no-cfg", action="store_true")
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    m = Roest(args.model, dtype=dtype, s3gen=args.s3gen, cfg=not args.no_cfg)
    text = "Et øjeblik, jeg kigger lige efter."
    for _ in m.stream(text, first=args.first, chunk=args.chunk, prompt_tokens=args.prompt_tokens):  # captures graphs
        pass
    voc = m.voc
    pre = timeit(lambda: m.prefill(text), 30)
    m.prefill(text)

    p0 = m.host_pos + 40  # a typical position mid-sentence
    L = next(b for b in m.buckets() if b > p0 + 50)

    def steps():
        m.pos.fill_(p0)
        m.n.fill_(1)
        for _ in range(50):
            m.graph[L].replay()

    step = timeit(steps, 10)
    step = {k: round(v / 50, 3) for k, v in step.items()}
    toks = torch.full((200,), 100, dtype=torch.long, device="cuda")
    w0 = timeit(lambda: voc._window(toks, 0, 0, args.first), 30)
    w1 = timeit(lambda: voc._window(toks, 40, voc.ctx, args.chunk), 30)
    res = dict(cfg=m.batch == 2, prefix_len=m.prefix_len, kv_bucket=L, prompt_tokens=args.prompt_tokens, model=args.model, s3gen=args.s3gen, dtype=args.dtype, prefill_ms=pre, step_ms=step,
               first_window_ms=w0, chunk_window_ms=w1,
               predicted_ttfa_min=round(pre["min"] + (args.first - 1) * step["min"] + w0["min"], 1),
               predicted_ttfa_p50=round(pre["p50"] + (args.first - 1) * step["p50"] + w0["p50"], 1),
               # audio seconds produced per wall second once streaming (25 tokens = 1 s)
               steady_rtf_min=round((args.chunk * step["min"] + w1["min"]) / (args.chunk - 3) / 40, 3))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
