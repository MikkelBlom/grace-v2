"""Acoustic replay tier: a "person" talks to the real, running Grace from a second speaker.

Grace runs as usual on her own speakers (the Z337); this script plays Danish utterances from
CoRal's conversation test split through ANOTHER output (default: the laptop's speakers), so the
real mic hears real room echo, real distance and real double-talk. Every playback is logged with
wall-clock times; `analyze` lines them up with the engine's trace.

ONLY run this while Mikkel is out of the house, never at night: it talks out loud for a long time.

  1. scripts/grace.ps1 -Memory data/replay-memory.db      (or: grace.exe --memory data/replay-memory.db)
  2. bench/.venv/Scripts/python.exe bench/sim/replay.py play --n 60 --device "Speakers (Realtek"
  3. bench/.venv/Scripts/python.exe bench/sim/replay.py analyze --log bench-results/replay/<run>.jsonl

Two kinds of trials: `turn` (the person speaks while Grace is quiet, then waits for her reply)
and `bargein` (the person starts talking while Grace is answering). CoRal is used for
recognition testing only (its licence bans speech synthesis).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_scenes import FS, active_rms, coral_utterances, db  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench-results" / "replay"


def device_index(name):
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"]
        if d["max_output_channels"] > 0 and "WASAPI" in api and name.lower() in d["name"].lower():
            return i
    sys.exit(f"no WASAPI output matching {name!r}; see python -m sounddevice")


def play(args):
    rng = random.Random(args.seed)
    dev = device_index(args.device)
    utts = coral_utterances(args.n, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"replay-{int(time.time())}.jsonl"
    print(f"playing {len(utts)} utterances on {sd.query_devices(dev)['name']}; log {log_path}")
    with open(log_path, "w", encoding="utf-8") as log:
        for i, (uid, text, audio) in enumerate(utts):
            kind = "bargein" if rng.random() < args.bargein else "turn"
            if kind == "bargein":
                # Ask something first so Grace is mid-reply when the person cuts in.
                _, _, opener = utts[(i + 1) % len(utts)]
                start = time.time()
                sd.play(opener * (db(args.level) / active_rms(opener)), FS, device=dev, blocking=True)
                log.write(json.dumps({"id": uid + "-opener", "kind": "opener", "start": start,
                                      "end": time.time()}) + "\n")
                time.sleep(rng.uniform(2.5, 4.0))  # Grace's reply has started
            pcm = audio * (db(args.level) / active_rms(audio))
            start = time.time()
            sd.play(pcm, FS, device=dev, blocking=True)
            end = time.time()
            log.write(json.dumps({"id": uid, "kind": kind, "text": text, "start": start,
                                  "end": end}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{i + 1}/{len(utts)} {kind}: {text[:60]}")
            time.sleep(args.gap)  # Grace answers; the next trial starts in silence


def words(s):
    return "".join(c.lower() if c.isalnum() else " " for c in s).split()


def word_error_rate(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def analyze(args):
    trials = [json.loads(l) for l in open(args.log, encoding="utf-8")]
    traces = sorted((ROOT / "traces").glob("engine-*.jsonl"))
    events = []
    for t in traces[-args.traces:]:
        for line in open(t, encoding="utf-8"):
            e = json.loads(line)
            if e.get("kind") == "event" and e.get("name") in ("transcript", "reply interrupted"):
                events.append(e)
    rows, interrupted = [], 0
    for tr in trials:
        if tr["kind"] == "opener":
            continue
        heard = [e for e in events if e["name"] == "transcript"
                 and tr["start"] - 0.5 <= e["ts_us"] / 1e6 <= tr["end"] + 6.0]
        text = " ".join(e["fields"].get("text", "") for e in heard
                        if not e["fields"].get("hallucinated"))
        wer = word_error_rate(tr["text"], text) if text else 1.0
        rows.append((tr, text, wer))
        if tr["kind"] == "bargein" and any(e["name"] == "reply interrupted"
                                           and tr["start"] <= e["ts_us"] / 1e6 <= tr["end"] + 2
                                           for e in events):
            interrupted += 1
    for kind in ("turn", "bargein"):
        sel = [r for r in rows if r[0]["kind"] == kind]
        if sel:
            missed = sum(1 for r in sel if not r[1])
            print(f"{kind}: {len(sel)} trials, missed {missed}, mean WER {np.mean([r[2] for r in sel]):.3f}")
    n_barge = sum(1 for r in rows if r[0]["kind"] == "bargein")
    if n_barge:
        print(f"barge-ins that stopped Grace: {interrupted}/{n_barge}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--device", default="Speakers (Realtek")
    p.add_argument("--level", type=float, default=-20.0, help="playback level, dBFS active")
    p.add_argument("--bargein", type=float, default=0.3, help="share of barge-in trials")
    p.add_argument("--gap", type=float, default=10.0, help="seconds between trials")
    p.add_argument("--seed", type=int, default=21)
    a = sub.add_parser("analyze")
    a.add_argument("--log", required=True)
    a.add_argument("--traces", type=int, default=1, help="how many latest engine traces to read")
    args = ap.parse_args()
    {"play": play, "analyze": analyze}[args.cmd](args)
