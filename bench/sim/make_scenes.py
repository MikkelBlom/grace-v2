"""Builds grace-sim scenes from public corpora: what the laptop mic would hear in a room.

  bench/.venv/Scripts/python.exe bench/sim/make_scenes.py --set far --n 200

Sets (each a directory under fixtures/corpora/scenes/ with scenes.jsonl + 48 kHz WAV pairs):
  clean       the corpus recordings as they are (the baseline)
  near        a person 0.4-0.8 m from the laptop, quiet room
  far         1.5-3 m away, room noise 10-20 dB below the speech
  doubletalk  Grace is talking when the person starts (barge-in), person 0.8-2 m away
  echo        only Grace talking (must trigger nothing)
  media       music / talk played through the speakers (must trigger nothing)
  noise       room sounds at the mic, nobody talking (must not become words)

People come from the CoRal v3 conversation test split (recognition use only; its licence bans
speech synthesis). Rooms are simulated with pyroomacoustics; noise, music and media talk come
from MUSAN (CC BY 4.0). Grace's voice is her real TTS voice, fetched once from the speech server.
"""

import argparse
import base64
import io
import json
import random
import socket
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parents[2]
CORAL = ROOT / "fixtures" / "coral-v3" / "conversation"
MUSAN = ROOT / "fixtures" / "corpora" / "musan"
GRACE = ROOT / "fixtures" / "corpora" / "grace-voice"
OUT = ROOT / "fixtures" / "corpora" / "scenes"
FS = 48_000
DELAY_MS = 120.0  # speaker → mic delay baked into the echo (the frontend starts calibrated)
# Echo levels are calibrated to real Z337 recordings (bench-results/aec): the laptop mic hears
# Grace about 3 dB below the loopback signal (median), 6-7 dB below at the 20th percentile.

GRACE_LINES = [
    "Sure, that's the one with the red cover, I think it's on the shelf behind you.",
    "Honestly, I'd go with the second option, it's cheaper and you won't miss the extra features.",
    "Your sister's flight lands at four, so you have time to finish the report first.",
    "That's a classic off by one error, the loop runs one time too many.",
    "I don't know that yet, but I can look it up once I have a search tool.",
    "Okay, short version: Rust is fast, safe, and a bit stubborn about lifetimes.",
    "Three things are due this week, and the database assignment is the scariest one.",
    "You asked me to remind you about the semester project details, so here I am.",
    "Coffee number three already? Bold strategy, let's see if it pays off.",
    "The meeting moved to Thursday at ten, same room as last time.",
    "If you want, I can read the first paragraph out loud, it's not too long.",
    "I live in your laptop, so my day mostly consisted of listening to the fan.",
    "That sounds like a good plan, but you might want to test it on a copy first.",
    "Nope, that one's on me, I misheard you. Try saying it again?",
    "The weather looks grey all week, so maybe skip the bike tomorrow.",
    "Let me put it differently: the cache only helps if the prefix stays the same.",
    "Right, so the exam was cancelled, which frees up your whole December.",
    "Good question. The short answer is no, and the long answer is also no.",
    "I'd split it into two steps, record first, then fix the transcripts.",
    "Forty two minutes, if the trains are on time, which they rarely are.",
]


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def active_rms(x, rate=FS):
    """RMS over the loudest half of 20 ms frames, so pauses don't count."""
    n = rate // 50
    frames = x[: len(x) // n * n].reshape(-1, n)
    e = np.sort(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)))
    return float(e[len(e) // 2 :].mean() + 1e-12)


def db(v):
    return 10 ** (v / 20)


def load(path_or_bytes, rate=FS):
    a, r = sf.read(path_or_bytes, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if r != rate:
        g = np.gcd(r, rate)
        a = resample_poly(a, rate // g, r // g).astype(np.float32)
    return a


def coral_utterances(n, rng):
    rows = []
    for shard in sorted(CORAL.glob("test-*.parquet")):
        t = pq.read_table(shard, columns=["id_speaker", "text", "overlap", "audio"])
        for spk, text, overlap, audio in zip(*(t.column(c).to_pylist() for c in
                                                 ["id_speaker", "text", "overlap", "audio"])):
            if text and text.strip() and not overlap:
                rows.append((spk, text.strip(), audio["bytes"]))
    rng.shuffle(rows)
    per_speaker, picked = {}, []
    for spk, text, data in rows:
        if per_speaker.get(spk, 0) >= max(3, n // 20):
            continue
        a = load(io.BytesIO(data))
        if not 1.5 <= len(a) / FS <= 12.0:
            continue
        per_speaker[spk] = per_speaker.get(spk, 0) + 1
        picked.append((f"coral-{spk}-{len(picked):04d}", text, a))
        if len(picked) == n:
            break
    return picked


def grace_voice():
    """Grace's real voice (Pocket TTS via the speech server), cached as 48 kHz WAVs."""
    GRACE.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, line in enumerate(GRACE_LINES):
        path = GRACE / f"grace-{i:02d}.wav"
        if not path.exists():
            with socket.create_connection(("127.0.0.1", 7811)) as s:
                f = s.makefile("rwb")
                f.write((json.dumps({"id": 1, "text": line}) + "\n").encode())
                f.flush()
                pcm, rate = [], 24000
                for msg in f:
                    m = json.loads(msg)
                    if m.get("done") or m.get("error"):
                        break
                    rate = m["rate"]
                    pcm.append(np.frombuffer(base64.b64decode(m["pcm"]), dtype="<f4"))
            sf.write(path, np.concatenate(pcm), rate)
        clips.append(load(path))
    return clips


def musan(kind):
    return sorted((MUSAN / kind).rglob("*.wav")) if (MUSAN / kind).exists() else []


DEMAND = ROOT / "fixtures" / "corpora" / "demand"
ESC50 = ROOT / "fixtures" / "corpora" / "esc50"
FLEURS = ROOT / "fixtures" / "stt" / "fleurs"
# Sounds a person makes near a laptop without talking: these must never become words.
ESC_EVENTS = {"breathing", "keyboard_typing", "coughing", "sneezing", "mouse_click",
              "door_wood_knock", "drinking_sipping", "clock_tick", "footsteps",
              "door_wood_creaks", "can_opening", "glass_breaking", "clapping", "snoring",
              "vacuum_cleaner", "washing_machine", "water_drops", "pouring_water"}


def room_beds():
    """Steady room noise: DEMAND indoor recordings (CC BY 4.0), else MUSAN noise."""
    indoor = [p for p in sorted(DEMAND.glob("*.wav"))
              if any(k in p.stem.upper() for k in ("KITCHEN", "LIVING", "WASHING", "OFFICE",
                                                     "MEETING", "HALLWAY", "CAFE"))]
    return indoor or sorted(DEMAND.glob("*.wav")) or musan("noise")


def esc_events():
    meta = ESC50 / "meta" / "esc50.csv"
    if not meta.exists():
        return []
    import csv

    with open(meta, encoding="utf-8") as f:
        return [ESC50 / "audio" / r["filename"] for r in csv.DictReader(f)
                if r["category"] in ESC_EVENTS]


def media_sources():
    """Talk and music a PC plays: Danish speech (FLEURS, the hardest case) and MUSAN music."""
    return sorted(FLEURS.glob("*.wav")) + musan("music")


class Room:
    """A random shoebox room with the laptop mic on a desk."""

    def __init__(self, rng):
        self.dims = [rng.uniform(3.0, 5.5), rng.uniform(3.0, 6.0), rng.uniform(2.4, 2.9)]
        rt60 = rng.uniform(0.25, 0.6)
        self.absorption, self.order = pra.inverse_sabine(rt60, self.dims)
        self.mic = np.array([self.dims[0] / 2, 0.8, 0.85])
        self.rng = rng

    def rir(self, distance, height=1.2):
        ang = self.rng.uniform(-1.0, 1.0)
        src = self.mic + np.array([distance * np.sin(ang), distance * np.cos(ang), 0.0])
        src[2] = height
        src = np.clip(src, 0.1, np.array(self.dims) - 0.1)
        room = pra.ShoeBox(self.dims, fs=FS, materials=pra.Material(self.absorption),
                           max_order=min(self.order, 17))
        room.add_source(src)
        room.add_microphone(self.mic)
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float32)
        # Remove pyroomacoustics' fixed filter delay so speech timing stays as labelled, and
        # scale so the direct path has amplitude 1/r (unity at 1 m): level falls with distance
        # while the room's reverb keeps its shape.
        peak = int(np.argmax(np.abs(h)))
        h = h[max(0, peak - int(distance / 343 * FS)) :]
        return h / float(np.max(np.abs(h))) / max(distance, 0.1)


def at_mic(clean, h, level_db):
    """Sound through the room, scaled so its direct sound 1 m from the mic is `level_db` dBFS."""
    wet = fftconvolve(clean, h)[: len(clean) + FS // 2].astype(np.float32)
    return wet * (db(level_db) / active_rms(clean))


def noise_bed(n, rng, files):
    if files:
        out = np.zeros(n, np.float32)
        pos = 0
        while pos < n:
            a = load(rng.choice(files))
            start = rng.randrange(max(1, len(a) - (n - pos)))
            take = min(len(a) - start, n - pos)
            out[pos : pos + take] = a[start : start + take]
            pos += take
        return out
    # Fallback until MUSAN is downloaded: low-passed white noise, a rough room hum.
    w = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(n)
    return np.convolve(w, np.ones(8) / 8, "same").astype(np.float32)


def speaker_out(far):
    """What the speakers emit for a digital signal: a little soft clipping, like cheap drivers."""
    return np.tanh(1.5 * far) / 1.5


WARMUP_S = 6.0  # Grace talking before each scene, so the echo canceller has converged as live


class EchoPath:
    """One scene's speaker → mic path: room response, delay, cheap-driver distortion, gain.

    The same path is used for the warm-up and the scene, as in a real room. Gain is set so the
    mic hears Grace about 3 dB below the loopback (median of real Z337 recordings).
    """

    def __init__(self, room, rng):
        self.h = room.rir(rng.uniform(0.3, 0.8), 0.9)
        self.gain = db(rng.uniform(-14.0, -8.0))

    def __call__(self, far, total, at):
        """Adds the echo of `far` (played from sample `at`) to a zero mic track of `total`."""
        mic = np.zeros(total, np.float32)
        wet = fftconvolve(speaker_out(far), self.h)[: len(far) + FS // 2] * self.gain
        e0 = at + int(DELAY_MS / 1000 * FS)
        if e0 < total:
            mic[e0 : e0 + len(wet)] += wet[: total - e0]
        return mic


def with_warmup(mic, loop, row, echo, grace, rng, noises):
    """Prepends WARMUP_S of Grace talking (and its echo) and shifts the scene's labels."""
    far = np.concatenate([rng.choice(grace) for _ in range(4)])[: int((WARMUP_S - 1) * FS)]
    far = far * db(-20.0) / active_rms(far)
    w = int(WARMUP_S * FS)
    wl = np.zeros(w, np.float32)
    wl[: len(far)] = far
    wm = echo(far, w, 0)
    bed = noise_bed(w, rng, noises)
    wm += bed / rms(bed) * db(-60.0)
    row["user"] = [[a + WARMUP_S, b + WARMUP_S] for a, b in row["user"]]
    row["eval_from"] = WARMUP_S
    return np.concatenate([wm, mic]), np.concatenate([wl, loop]), row


def build(set_name, n, seed):
    rng = random.Random(seed)
    out = OUT / set_name
    out.mkdir(parents=True, exist_ok=True)
    noises = room_beds()
    grace = grace_voice()
    rows = []

    def save(sid, mic, loop, row, echo):
        mic, loop, row = with_warmup(mic, loop, row, echo, grace, rng, noises)
        peak = max(1.0, float(np.max(np.abs(mic))) / 0.98)
        sf.write(out / f"{sid}.mic.wav", (mic / peak).astype(np.float32), FS, subtype="FLOAT")
        sf.write(out / f"{sid}.loop.wav", loop.astype(np.float32), FS, subtype="FLOAT")
        rows.append(row)

    if set_name in ("clean", "near", "far", "doubletalk"):
        for sid, text, clean in coral_utterances(n, rng):
            room = Room(rng)
            echo = EchoPath(room, rng)
            dist = {"clean": (0.0, 0.0), "near": (0.4, 0.8), "far": (1.5, 3.0),
                    "doubletalk": (0.8, 2.0)}[set_name]
            d = rng.uniform(*dist)
            # "clean" is the corpus audio as recorded: the baseline the room sets are judged by.
            user = clean * (db(-28.0) / active_rms(clean)) if set_name == "clean" else at_mic(
                clean, room.rir(d), -28.0)
            lead = rng.uniform(0.8, 1.5) + (2.0 if set_name == "doubletalk" else 0.0)
            total = int((lead + len(clean) / FS + 2.5) * FS)
            mic = np.zeros(total, np.float32)
            loop = np.zeros(total, np.float32)
            s0 = int(lead * FS)
            mic[s0 : s0 + len(user)] += user[: total - s0]
            if set_name != "clean":
                snr = {"near": (25, 35), "far": (10, 20), "doubletalk": (15, 25)}[set_name]
                bed = noise_bed(total, rng, noises)
                mic += bed / rms(bed) * active_rms(user) * db(-rng.uniform(*snr))
            if set_name == "doubletalk":
                # Grace starts 2 s before the person and keeps talking for 0.5-1.5 s after they
                # start (in the live engine the barge-in stops her).
                far = np.concatenate([rng.choice(grace) for _ in range(3)])
                g0 = s0 - 2 * FS
                g1 = min(total, s0 + int(rng.uniform(0.5, 1.5) * FS), g0 + len(far))
                far = far[: g1 - g0] * db(-20.0) / active_rms(far)
                loop[g0:g1] = far
                mic += echo(far, total, g0)
            save(sid, mic, loop, {"id": sid, "kind": "speech", "text": text,
                                  "user": [[lead, lead + len(clean) / FS]], "delay_ms": DELAY_MS,
                                  "distance_m": round(d, 2)}, echo)
    elif set_name in ("echo", "media"):
        sources = media_sources() if set_name == "media" else None
        for i in range(n):
            room = Room(rng)
            echo = EchoPath(room, rng)
            if set_name == "echo":
                far = np.concatenate([rng.choice(grace) for _ in range(3)])[: 12 * FS]
            else:
                far = np.concatenate([load(rng.choice(sources)) for _ in range(3)])[: 12 * FS]
            far = far * db(rng.uniform(-24.0, -16.0)) / active_rms(far)
            total = len(far) + 2 * FS
            loop = np.zeros(total, np.float32)
            loop[FS : FS + len(far)] = far
            mic = echo(far, total, FS)
            bed = noise_bed(total, rng, noises)
            mic += bed / rms(bed) * db(-60.0)
            sid = f"{set_name}-{i:04d}"
            save(sid, mic, loop, {"id": sid, "kind": set_name, "text": "", "user": [],
                                  "delay_ms": DELAY_MS}, echo)
    elif set_name == "noise":
        events = esc_events()
        assert events, "noise needs ESC-50: bench/fetch_corpora.py esc50"
        for i in range(n):
            # Two or three 5 s event clips in a row over a quiet room bed.
            a = np.concatenate([load(rng.choice(events)) for _ in range(rng.randint(2, 3))])
            room = Room(rng)
            echo = EchoPath(room, rng)
            wet = at_mic(a, room.rir(rng.uniform(0.3, 2.0)), rng.uniform(-40.0, -22.0))
            total = len(wet) + 2 * FS
            mic = np.zeros(total, np.float32)
            mic[FS : FS + len(wet)] = wet
            if noises:
                bed = noise_bed(total, rng, noises)
                mic += bed / rms(bed) * db(-55.0)
            sid = f"noise-{i:04d}"
            save(sid, mic, np.zeros(total, np.float32),
                 {"id": sid, "kind": "noise", "text": "", "user": [], "delay_ms": DELAY_MS}, echo)
    with open(out / "scenes.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{set_name}: {len(rows)} scenes in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    choices=["clean", "near", "far", "doubletalk", "echo", "media", "noise"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    build(a.set, a.n, a.seed)
