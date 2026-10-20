"""Builds turn-taking scenes for `grace-turn-sim`: one person's turn, possibly paused mid-way.

  bench/.venv/Scripts/python.exe bench/sim/make_turns.py --n-multi 150 --n-single 100

Writes fixtures/corpora/scenes/turns/ (scenes.jsonl + 48 kHz mono `<id>.mic.wav`).

People are real Danish speech from the CoRal v3 conversation test split (24 unseen speakers;
recognition use only). A **multi** scene is 2-3 consecutive segments by the same speaker (CoRal
splits a speaker's turn where they pause), joined with a pause drawn from a log-normal
(median 0.9 s, 0.35-3.0 s): a person who stops, thinks, and goes on. A **single** scene is one
segment that the other speaker answered (a finished turn, >= 3 words, not ending on a hanging
word). The mic is a close clip mic (the DJI headset setup): the speech as recorded plus a
DEMAND room bed 25-35 dB below it. Each part is trimmed to its voiced span, so `parts` holds
the true speech edges and `true_end` is when the person really stopped.
"""

import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from make_scenes import FS, active_rms, db, load, noise_bed, room_beds  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORAL = ROOT / "fixtures" / "coral-v3" / "conversation"
OUT = ROOT / "fixtures" / "corpora" / "scenes" / "turns"
HANG = {"øh", "øhm", "eh", "ehm", "hm", "hmm", "mm", "æh", "og", "men", "eller", "så", "at",
        "fordi", "hvis", "når", "som", "der", "altså", "til", "i", "på", "med", "for", "om", "den",
        "det", "en", "et"}
LEAD, TAIL = 1.5, 8.0


def trim(a, rel_db=-35.0, margin=0.05):
    """Cuts leading/trailing silence: 10 ms frames below `rel_db` of the loudest frame."""
    n = FS // 100
    frames = a[: len(a) // n * n].reshape(-1, n)
    e = 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
    on = np.where(e > e.max() + rel_db)[0]
    if len(on) == 0:
        return a[:0]
    m = int(margin * FS)
    return a[max(0, on[0] * n - m): min(len(a), (on[-1] + 1) * n + m)]


def rows():
    """CoRal test rows grouped by conversation, in segment order."""
    convs = {}
    for shard in sorted(CORAL.glob("test-*.parquet")):
        t = pq.read_table(shard, columns=["id_conversation", "id_speaker", "overlap", "text", "audio"])
        for cid, spk, ov, text, audio in zip(*(t.column(c).to_pylist() for c in
                                                 ["id_conversation", "id_speaker", "overlap", "text", "audio"])):
            conv, idx = cid.rsplit("_", 1)
            convs.setdefault(conv, {})[int(idx)] = (spk, ov, (text or "").strip(), audio["bytes"])
    return convs


def last_word(text):
    words = re.findall(r"\w+", text.lower())
    return words[-1] if words else ""


def candidates(convs):
    multi, single = [], []
    for conv, segs in convs.items():
        for i, (spk, ov, text, _) in segs.items():
            if ov or not text:
                continue
            run = [i]
            while len(run) < 3 and (j := run[-1] + 1) in segs and segs[j][0] == spk \
                    and not segs[j][1] and segs[j][2]:
                run.append(j)
            prev_same = (i - 1) in segs and segs[i - 1][0] == spk
            if len(run) >= 2 and not prev_same:
                multi.append((conv, run))
            nxt = segs.get(i + 1)
            if (nxt is None or nxt[0] != spk) and len(text.split()) >= 3 \
                    and last_word(text) not in HANG and not prev_same:
                single.append((conv, [i]))
    return multi, single


def build(n_multi, n_single, seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    convs = rows()
    multi, single = candidates(convs)
    rng.shuffle(multi)
    rng.shuffle(single)
    beds = room_beds()
    OUT.mkdir(parents=True, exist_ok=True)
    out = open(OUT / "scenes.jsonl", "w", encoding="utf-8")
    per_speaker = {}
    made = {"multi": 0, "single": 0}
    want = {"multi": n_multi, "single": n_single}
    for kind, pool in (("multi", multi), ("single", single)):
        for conv, run in pool:
            if made[kind] >= want[kind]:
                break
            spk = convs[conv][run[0]][0]
            if per_speaker.get((kind, spk), 0) >= max(3, want[kind] // 12):
                continue
            parts = [trim(load(io.BytesIO(convs[conv][i][3]))) for i in run]
            if any(not 0.8 <= len(p) / FS <= 10.0 for p in parts):
                continue
            # Keep only as many parts as fit in 25 s of speech.
            while sum(len(p) for p in parts) / FS > 25.0 and len(parts) > 2:
                parts.pop()
            if kind == "multi" and len(parts) < 2:
                continue
            texts = [convs[conv][i][2] for i in run[: len(parts)]]
            gaps = [float(np.clip(nrng.lognormal(np.log(0.9), 0.55), 0.35, 3.0))
                    for _ in parts[1:]]
            level = -26.0 + rng.uniform(-4, 4)
            edges, pos = [], LEAD
            pieces = [np.zeros(int(LEAD * FS), np.float32)]
            for k, p in enumerate(parts):
                p = p * (db(level) / active_rms(p))
                edges.append([round(pos, 3), round(pos + len(p) / FS, 3)])
                pieces.append(p.astype(np.float32))
                pos += len(p) / FS
                if k < len(gaps):
                    pieces.append(np.zeros(int(gaps[k] * FS), np.float32))
                    pos += int(gaps[k] * FS) / FS
            pieces.append(np.zeros(int(TAIL * FS), np.float32))
            mic = np.concatenate(pieces)
            snr = rng.uniform(25, 35)
            bed = noise_bed(len(mic), rng, beds)
            mic = mic + bed * (db(level - snr) / (active_rms(bed) + 1e-9))
            sid = f"{kind}-{made[kind]:04d}"
            sf.write(OUT / f"{sid}.mic.wav", np.clip(mic, -1, 1), FS, subtype="FLOAT")
            out.write(json.dumps({"id": sid, "kind": kind, "speaker": spk, "texts": texts,
                                  "parts": edges, "gaps": [round(g, 3) for g in gaps],
                                  "true_end": edges[-1][1]}, ensure_ascii=False) + "\n")
            per_speaker[(kind, spk)] = per_speaker.get((kind, spk), 0) + 1
            made[kind] += 1
    out.close()
    print(f"{made} scenes in {OUT} (candidates: {len(multi)} multi, {len(single)} single)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-multi", type=int, default=150)
    ap.add_argument("--n-single", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(a.n_multi, a.n_single, a.seed)
