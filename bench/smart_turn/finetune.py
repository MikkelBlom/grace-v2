"""Fine-tune Smart Turn v3.2 on Danish while replaying the multilingual v3.2 train data.

Uses the official model class, loss (BCE with per-batch pos_weight), metrics and ONNX export from
pipecat-ai/smart-turn train.py (BSD-2, cloned to fixtures/corpora/smart-turn/code), starting from
the published v3.2 weights recovered by onnx_to_torch.py. Differences from the official loop, and
why: features are computed on the GPU in batches by a verified port of the official extractor
(the numpy path costs ~250 ms per clip here), and data streams from parquet with a weighted mix so
Danish can be up-weighted without copying the 41 GB set.

  python bench/smart_turn/finetune.py --run da-v1 --steps 4000 --lr 3e-5 --da-weight 0.5

Whole pipeline (2026-10-11 candidate = da-v1 interpolated 70 % with the base, "wise70"):
  download.py -> onnx_to_torch.py -> build_coral.py (conv test/val/train, read_aloud train)
  -> eval_onnx.py / eval_harsh.py (baseline) -> finetune.py -> wise.py <run>/last 0.7
  -> eval_onnx.py / eval_harsh.py (candidate) -> turn_any + rust_check.py (Rust path)
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "fixtures/corpora/smart-turn/code")
os.environ.setdefault("WANDB_MODE", "disabled")
import common  # noqa: E402
from train import SmartTurnV3Model, export_to_onnx_fp32, CONFIG as OFFICIAL  # noqa: E402

ST = "fixtures/corpora/smart-turn"


class ParquetStream(IterableDataset):
    """Endless shuffled stream of (8 s window, label) from parquet row groups; each worker reads a
    disjoint, reshuffled set of row groups and shuffles inside a buffer."""

    def __init__(self, files, seed, holdout=None, buffer=256, filt=None, noise=0.0):
        self.units = []
        for f in files:
            for rg in range(pq.ParquetFile(f).num_row_groups):
                self.units.append((f, rg))
        self.seed, self.buffer, self.holdout, self.filt = seed, buffer, holdout, filt
        self.noise_p, self.bank = noise, None

    def unit_rows(self, f, rg):
        cols = ["audio", "endpoint_bool"] + (["id"] if self.holdout else [])
        t = pq.ParquetFile(f).read_row_group(rg, columns=cols).to_pydict()
        for i in range(len(t["endpoint_bool"])):
            if self.holdout and self.holdout(t["id"][i]):
                continue
            yield t["audio"][i], bool(t["endpoint_bool"][i])

    def rows(self, rng, open_units=6):
        """Interleave rows from several open row groups, so a batch is not one conversation."""
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        units = self.units[wid::nw] or self.units
        epoch = 0
        while True:
            order = list(units)
            random.Random(self.seed * 1000 + wid * 7 + epoch).shuffle(order)
            pending = iter(order)
            live = []
            while True:
                while len(live) < open_units:
                    nxt = next(pending, None)
                    if nxt is None:
                        break
                    live.append(self.unit_rows(*nxt))
                if not live:
                    break
                k = rng.randrange(len(live))
                row = next(live[k], None)
                if row is None:
                    live.pop(k)
                    continue
                yield row
            epoch += 1

    def __iter__(self):
        rng = random.Random(self.seed)
        buf = []
        for audio, label in self.rows(rng):
            buf.append((audio, label))
            if len(buf) >= self.buffer:
                j = rng.randrange(len(buf))
                buf[j], buf[-1] = buf[-1], buf[j]
                a, y = buf.pop()
                x = common.decode(a)
                if self.noise_p and rng.random() < self.noise_p:
                    x = self.add_noise(x, rng)
                yield common.last_8s(x).astype(np.float32), y

    def add_noise(self, x, rng):
        """v3.2-style background noise, same for both labels: MUSAN/DEMAND at 5-25 dB SNR."""
        if self.bank is None:
            import build_coral
            self.bank = build_coral.load_noise_bank(max_files=120, seed=self.seed)
            self.bank = [b[: 30 * common.SR] for b in self.bank]
        if not self.bank:
            return x
        nz = self.bank[rng.randrange(len(self.bank))]
        if len(nz) < len(x):
            nz = np.tile(nz, len(x) // len(nz) + 1)
        st = rng.randrange(len(nz) - len(x) + 1)
        nz = nz[st: st + len(x)]
        ps = float((x ** 2).mean()) + 1e-10
        pn = float((nz ** 2).mean()) + 1e-10
        y = x + nz * np.sqrt(ps / (pn * 10 ** (rng.uniform(5, 25) / 10)))
        peak = float(np.abs(y).max())
        return (y * (0.99 / peak) if peak > 0.99 else y).astype(np.float32)


class Mix(IterableDataset):
    def __init__(self, streams, weights, seed):
        self.streams, self.weights, self.seed = streams, weights, seed

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0))
        its = [iter(s) for s in self.streams]
        while True:
            k = rng.choices(range(len(its)), self.weights)[0]
            yield next(its[k])


def collate(b):
    return torch.from_numpy(np.stack([x for x, _ in b])), torch.tensor([y for _, y in b], dtype=torch.long)


def in_v32_holdout(uid):
    """Stable 3 % of the v3.2 train rows held out for validation (by id hash)."""
    return (hash_str(uid) % 100) < 3


def hash_str(s):
    import zlib
    return zlib.crc32(s.encode())


@torch.no_grad()
def evaluate(model, tf, files, device, limit=None, holdout_only=False):
    model.eval()
    probs, labels = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["audio", "endpoint_bool", "id"]).to_pydict()
            idx = [i for i in range(len(t["id"])) if not holdout_only or in_v32_holdout(t["id"][i])]
            for s in range(0, len(idx), 128):
                part = idx[s:s + 128]
                w = np.stack([common.last_8s(common.decode(t["audio"][i])) for i in part]).astype(np.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(tf(torch.from_numpy(w)))["logits"].float().cpu().numpy()[:, 0]
                probs.append(p)
                labels += [t["endpoint_bool"][i] for i in part]
            if limit and len(labels) >= limit:
                break
        if limit and len(labels) >= limit:
            break
    model.train()
    return common.metrics(np.array(labels), np.concatenate(probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", default="bench-results/smart_turn/base_torch")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--da-weight", type=float, default=0.4, help="share of each batch drawn from Danish CoRal/synthetic")
    ap.add_argument("--da", default=f"{ST}/da-coral/conv-train/*.parquet,{ST}/da-coral/read-train/*.parquet,{ST}/da-synth/train/*.parquet")
    ap.add_argument("--da-mix", default="0.75,0.25,0.15", help="weights within Danish, per --da glob (missing ones dropped)")
    ap.add_argument("--noise", type=float, default=0.4, help="probability of background noise on Danish clips")
    ap.add_argument("--ml-tol", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mem-gb", type=float, default=7.5)
    a = ap.parse_args()

    device = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb / total))
    out = f"bench-results/smart_turn/runs/{a.run}"
    os.makedirs(out, exist_ok=True)
    log = open(f"{out}/log.jsonl", "a")

    replay = sorted(glob.glob(f"{ST}/v3.2-train/data/*.parquet"))
    da_globs = a.da.split(",")
    da_w = [float(x) for x in a.da_mix.split(",")]
    da_sets = [(sorted(glob.glob(g)), w) for g, w in zip(da_globs, da_w)]
    da_sets = [(f, w) for f, w in da_sets if f]
    print("replay shards", len(replay), "| danish:", [(len(f), w) for f, w in da_sets], flush=True)
    streams = [ParquetStream(replay, 1, holdout=in_v32_holdout)] + [ParquetStream(f, 10 + i, noise=a.noise) for i, (f, _) in enumerate(da_sets)]
    dw = sum(w for _, w in da_sets)
    weights = [1 - a.da_weight] + [a.da_weight * w / dw for _, w in da_sets]
    loader = DataLoader(Mix(streams, weights, 42), batch_size=a.batch, num_workers=a.workers,
                        collate_fn=collate, persistent_workers=True, prefetch_factor=2)

    model = SmartTurnV3Model.from_pretrained(a.init).to(device).train()
    tf = common.TorchFeatures(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=OFFICIAL["weight_decay"])
    warm = int(a.warmup * a.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    val_da = sorted(glob.glob(f"{ST}/da-coral/conv-val/*.parquet"))
    best = -1.0
    m0 = evaluate(model, tf, val_da, device) if val_da else {"acc": float("nan")}
    base_ml = evaluate(model, tf, replay[:12], device, holdout_only=True)["acc"]
    print(f"start: coral-val acc {m0['acc']:.3f} | v3.2-holdout acc {base_ml:.3f}", flush=True)
    t0 = time.time()
    it = iter(loader)
    for step in range(1, a.steps + 1):
        x, y = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = tf(x)
            outp = model(feats, labels=y.to(device))
        loss = outp["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.2f}s/step mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m_da = evaluate(model, tf, val_da, device) if val_da else {"acc": float("nan")}
            m_ml = evaluate(model, tf, replay[:12], device, holdout_only=True)
            rec = {"step": step, "val_da": m_da, "val_multi": m_ml}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"EVAL step {step}: coral-val acc {m_da['acc']:.3f} falseEnd {m_da.get('false_turn_end', float('nan')):.3f}"
                  f" | v3.2-holdout acc {m_ml['acc']:.3f} falseEnd {m_ml['false_turn_end']:.3f} (n={m_ml['n']})", flush=True)
            # select on Danish val, but never accept a multilingual drop > --ml-tol vs. the start.
            # The holdout comes from the v3.2 TRAIN set, which the base model was trained on, so its
            # start value is optimistic; the real regression check is the v3.2 test set afterwards.
            score = m_da["acc"] if m_ml["acc"] >= base_ml - a.ml_tol else -1
            if score > best:
                best = score
                model.save_pretrained(f"{out}/best")
                json.dump(rec, open(f"{out}/best/val.json", "w"), indent=1)
                print("  saved best", flush=True)
    model.save_pretrained(f"{out}/last")
    best_model = SmartTurnV3Model.from_pretrained(f"{out}/best" if best >= 0 else f"{out}/last").eval().cpu()
    onnx_path = export_to_onnx_fp32(best_model, f"{out}/model_fp32.onnx", OFFICIAL)
    print("exported", onnx_path, flush=True)


if __name__ == "__main__":
    main()
