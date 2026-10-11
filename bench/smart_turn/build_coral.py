"""Builds Danish Smart Turn clips from CoRal v3 (OpenRAIL-D: recognition/turn detection only, never
synthesis) in the smart-turn-data schema, written as parquet shards.

  python bench/smart_turn/build_coral.py --split conversation/train --out fixtures/corpora/smart-turn/da-coral/conv-train
  python bench/smart_turn/build_coral.py --split read_aloud/train  --out fixtures/corpora/smart-turn/da-coral/read-train
  python bench/smart_turn/build_coral.py --split conversation/test --out fixtures/corpora/smart-turn/da-coral/conv-test --no-noise

Labels (endpoint_bool = turn complete):
- complete: conversation segment followed by the other speaker (a real turn of >= 3 words, or a
  missing segment: CoRal splits speakers across train/val/test, so the partner's audio is
  elsewhere), no overlap, not ending in a filler/conjunction/preposition/article; read-aloud: the whole
  sentence.
- incomplete (cut): the same kind of segment cut at a word boundary found by CTC forced alignment
  (Røst wav2vec2), cut inside the gap between two words, never mid-word; at least one word removed.
- incomplete (natural): conversation segment followed by the SAME speaker that ends in a filler or
  hanging word ("øh", "og", "at", ...) - a real mid-turn pause.
Every clip ends at the last word's acoustic end (energy-based, from the CTC anchor), then gets the
same tail for both labels: 0.15-0.6 s of either digital silence or the clip's own noise floor.
Noise (optional): with p=0.5 MUSAN/DEMAND noise over the whole clip at 5-25 dB SNR, both labels.
"""
import argparse
import glob
import io
import os
import random
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

SR = 16000
FILLERS = {"øh", "øhm", "øhh", "øhhh", "eh", "ehm", "hm", "hmm", "mm", "mmm", "æh", "æhm"}
HANG = FILLERS | {
    "og", "men", "eller", "så", "at", "fordi", "hvis", "når", "som", "der", "altså", "til", "i", "på",
    "med", "af", "for", "om", "fra", "ved", "efter", "under", "over", "en", "et", "den", "det", "de",
    "min", "mit", "mine", "din", "dit", "sin", "jeg", "du", "vi", "han", "hun", "man", "er", "har",
    "var", "skal", "vil", "kan", "skulle", "ville", "kunne", "må", "bliver", "blev", "lige", "sådan",
    "nogle", "noget", "nogen", "meget", "mere", "hvor", "hvad", "hvordan", "hvorfor", "end", "også",
}
BACKCHANNEL_MAX_WORDS = 2


def last_word(text):
    w = [x.strip(".,!?;:\"'").lower() for x in text.split()]
    w = [x for x in w if x]
    return w[-1] if w else ""


def plan_conversation(files, rng):
    """Read metadata only; return a DataFrame of rows to use with their role."""
    meta = []
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        t = pf.read(columns=["id_conversation", "id_speaker", "text", "overlap"]).to_pandas()
        t["file"] = fi
        t["row"] = np.arange(len(t))
        meta.append(t)
    m = pd.concat(meta, ignore_index=True)
    m["conv"] = m.id_conversation.str.rsplit("_", n=1).str[0]
    m["idx"] = m.id_conversation.str.rsplit("_", n=1).str[1].astype(int)
    key = {(c, i): (s, t) for c, i, s, t in zip(m.conv, m.idx, m.id_speaker, m.text)}
    roles = []
    for c, i, s, ov, text in zip(m.conv, m.idx, m.id_speaker, m.overlap, m.text):
        nxt = key.get((c, i + 1))
        lw = last_word(text)
        nwords = len(text.split())
        role = None
        if not ov:
            # CoRal splits by speaker, so the partner's segments are often in another split: a
            # missing next segment means the partner spoke next. A short partner reply that is
            # present (backchannel, "ja") does not count as taking the turn.
            other_next = nxt is None or (nxt[0] != s and len(nxt[1].split()) > BACKCHANNEL_MAX_WORDS)
            if other_next and lw not in HANG:
                role = "turn_end"
            elif nxt is not None and nxt[0] == s and lw in HANG and nwords >= 3:
                role = "mid_pause"
        if role is None and not ov and nwords >= 4:
            role = "cut_only"
        roles.append(role)
    m["role"] = roles
    return m


def energy_end(a, anchor_s, limit_s):
    """Acoustic end of speech after a CTC anchor: first 30 ms run below floor + 8 dB, bounded."""
    hop = 160
    n = len(a) // hop
    if n < 3:
        return min(anchor_s, limit_s)
    fr = a[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-6)
    floor = np.percentile(db, 10)
    thr = floor + 8
    i = int(anchor_s * 100)
    lim = min(int(limit_s * 100), n)
    run = 0
    while i < lim:
        run = run + 1 if db[i] < thr else 0
        if run >= 3:
            return (i - 2) / 100
        i += 1
    return lim / 100


def finish(a, end_s, rng, noise_bank, noisy):
    """Trim at end_s with a 10 ms fade, append the label-independent tail, maybe add noise."""
    e = int(end_s * SR)
    x = a[:e].copy()
    f = min(160, len(x))
    if f:
        x[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    tail_n = int(rng.uniform(0.15, 0.6) * SR)
    if rng.random() < 0.4:
        tail = np.zeros(tail_n, np.float32)
    else:
        hop = 160
        n = len(a) // hop
        fr = a[: n * hop].reshape(n, hop)
        rms = np.sqrt((fr ** 2).mean(1))
        level = float(np.percentile(rms, 10)) if n else 1e-4
        tail = (np.random.default_rng(rng.randrange(1 << 30)).standard_normal(tail_n) * level).astype(np.float32)
    x = np.concatenate([x, tail])
    if noisy and noise_bank and rng.random() < 0.5:
        nz = noise_bank[rng.randrange(len(noise_bank))]
        if len(nz) < len(x):
            nz = np.tile(nz, len(x) // len(nz) + 1)
        st = rng.randrange(len(nz) - len(x) + 1)
        nz = nz[st: st + len(x)]
        ps = float((x ** 2).mean()) + 1e-10
        pn = float((nz ** 2).mean()) + 1e-10
        snr = rng.uniform(5, 25)
        x = x + nz * np.sqrt(ps / (pn * 10 ** (snr / 10)))
    peak = np.abs(x).max()
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32)


def choose_cut(words, rng, min_prefix_s=0.7):
    """Pick a boundary between word k and k+1 (k+1 < n): cut inside the gap, never mid-word."""
    n = len(words)
    cands = []
    for k in range(1, n - 1):  # keep >= 2 words, remove >= 1 word
        end_k = words[k][2]
        start_n = words[k + 1][1]
        if end_k < min_prefix_s:
            continue
        gap = start_n - end_k
        if gap < 0.04:
            continue
        cands.append((k, gap))
    if not cands:
        return None
    # prefer real pauses half the time (the hard case: user pauses mid-sentence)
    if rng.random() < 0.5:
        long = [c for c in cands if c[1] >= 0.15]
        if long:
            cands = long
    k, gap = cands[rng.randrange(len(cands))]
    return k


def load_noise_bank(max_files=400, seed=0):
    files = glob.glob("fixtures/corpora/musan/noise/**/*.wav", recursive=True)
    files += glob.glob("fixtures/corpora/demand/*.wav")
    r = random.Random(seed)
    r.shuffle(files)
    bank = []
    for f in files[:max_files]:
        try:
            a, sr = sf.read(f, dtype="float32", always_2d=True)
            a = a.mean(1)
            if sr != SR:
                import soxr
                a = soxr.resample(a, sr, SR).astype(np.float32)
            if len(a) > SR:
                bank.append(a[: 60 * SR])
        except Exception as e:  # unreadable file: report and skip
            print("noise skip", f, e)
    print(f"noise bank: {len(bank)} clips from {len(files)} files", flush=True)
    return bank


def encode(x):
    b = io.BytesIO()
    sf.write(b, x, SR, format="FLAC", subtype="PCM_16")
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="conversation/train, read_aloud/train, conversation/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-shard", type=int, default=100000)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--part", default="0/1", help="i/n: only shards with index % n == i (parallel builders)")
    a = ap.parse_args()
    import align
    rng = random.Random(a.seed)
    files = sorted(glob.glob(f"fixtures/coral-v3/{a.split}-*.parquet"))
    assert files, a.split
    os.makedirs(a.out, exist_ok=True)
    conv = a.split.startswith("conversation")
    if conv:
        plan = plan_conversation(files, rng)
        print(plan.role.value_counts(dropna=False).to_dict(), flush=True)
    noise = [] if a.no_noise else load_noise_bank()
    aligner = align.Aligner()
    stats = {"complete": 0, "cut": 0, "mid_pause": 0, "align_fail": 0, "no_cut": 0}
    part, nparts = (int(v) for v in a.part.split("/"))
    for fi, f in enumerate(files):
        out_path = f"{a.out}/{os.path.basename(f)}"
        if fi % nparts != part or os.path.exists(out_path):
            continue
        pf = pq.ParquetFile(f)
        if conv:
            p = plan[(plan.file == fi) & plan.role.notna()]
            roles = dict(zip(p.row, p.role))
        rows = []
        base = 0
        for rg in range(pf.num_row_groups):
            cols = ["id_conversation" if conv else None, "id_speaker", "text", "audio"]
            t = pf.read_row_group(rg, columns=[c for c in cols if c]).to_pandas()
            for j in range(len(t)):
                r = base + j
                role = roles.get(r) if conv else "read"
                if role is None or len(rows) >= a.max_per_shard:
                    continue
                text = t.text.iloc[j]
                audio = common.decode(t.audio.iloc[j])
                dur = len(audio) / SR
                if dur < 0.6 or dur > 20:
                    continue
                words = aligner.align(audio, text)
                if not words or np.mean([w[3] for w in words if w[0] not in FILLERS] or [0]) < a.min_score:
                    stats["align_fail"] += 1
                    continue
                uid = (t.id_conversation.iloc[j] if conv else f"read_{os.path.basename(f)}_{r}")
                spk = t.id_speaker.iloc[j]
                lw = words[-1][0]

                def add(x, label, kind, endfill):
                    rows.append({"audio": {"bytes": encode(x), "path": f"{uid}_{kind}.flac"},
                                 "id": f"coral_{uid}_{kind}", "language": "dan", "endpoint_bool": label,
                                 "midfiller": any(w[0] in FILLERS for w in words[:-1]),
                                 "endfiller": endfill, "synthetic": False,
                                 "dataset": f"coral-{a.split.replace('/', '-')}", "kind": kind,
                                 "speaker": spk, "text": text})

                if role in ("turn_end", "read"):
                    end = energy_end(audio, words[-1][2], dur)
                    add(finish(audio, end, rng, noise, not a.no_noise), True, "complete", False)
                    stats["complete"] += 1
                if role == "mid_pause":
                    end = energy_end(audio, words[-1][2], dur)
                    add(finish(audio, end, rng, noise, not a.no_noise), False, "midpause", lw in FILLERS)
                    stats["mid_pause"] += 1
                if role in ("turn_end", "read", "cut_only"):
                    # cut_only rows exist to balance: use them only while cuts lag completes
                    if role == "cut_only" and stats["cut"] + stats["mid_pause"] >= stats["complete"]:
                        continue
                    k = choose_cut(words, rng)
                    if k is None:
                        stats["no_cut"] += 1
                        continue
                    limit = words[k + 1][1] - 0.02
                    end = min(energy_end(audio, words[k][2], limit), limit)
                    add(finish(audio, end, rng, noise, not a.no_noise), False, "cut", words[k][0] in FILLERS)
                    stats["cut"] += 1
            base += len(t)
        if rows:
            df = pd.DataFrame(rows)
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
        print(f"{os.path.basename(f)}: {len(rows)} clips  {stats}", flush=True)


if __name__ == "__main__":
    main()
