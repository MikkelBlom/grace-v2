"""Fetch candidate female prompt voices from kyutai/tts-voices and measure pitch (CPU, low priority).

  python kyutai_voices.py fetch      # downloads into fixtures/corpora/voices/kyutai/
  python kyutai_voices.py analyse    # writes fixtures/corpora/voices/kyutai/pitch.json

Only sources whose speakers consented to their voice being used for speech synthesis are fetched
(see docs/MODELS.md, "Female voices for Røst"):
- voice-donations/: Unmute Voice Donation Project, CC0. Donors ticked consent boxes accepting
  publication under CC0 and that TTS models may generate speech in their voice (Terms of Use v1, s. 2, 5, 7).
- alba-mackenna/: voice actress Alba MacKenna's character voices, released CC BY 4.0 for Kyutai TTS.
- unmute-prod-website/ "own recordings": Kyutai's own recordings for their TTS demo, CC0.
VCTK, Expresso and EARS speakers (e.g. Pocket TTS "azelma", "eve") are NOT fetched: no documented
consent to voice synthesis (Expresso/EARS are also non-commercial).
"""

import ctypes
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/corpora/voices/kyutai"
REPO = "kyutai/tts-voices"
OWN = ["unmute-prod-website/default_voice.wav", "unmute-prod-website/developer-1.mp3",
       "unmute-prod-website/developpeuse-3.wav", "unmute-prod-website/fabieng-enhanced-v2.wav"]


def low_priority():
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x40)  # IDLE_PRIORITY_CLASS
    except Exception as exc:  # not fatal: only politeness towards a live test
        print("could not lower priority:", exc, file=sys.stderr)


def files():
    from huggingface_hub import HfApi

    fs = HfApi().list_repo_files(REPO)
    don = sorted(f for f in fs if f.startswith("voice-donations/") and f.endswith("_enhanced.wav"))
    alba = sorted(f for f in fs if f.startswith("alba-mackenna/") and f.endswith(".wav"))
    return don + alba + [f for f in OWN if f in fs]


def fetch():
    from huggingface_hub import hf_hub_download

    for f in files():
        hf_hub_download(REPO, f, local_dir=OUT)
    print(len(files()), "files")


def analyse():
    import librosa
    import numpy as np

    res = {}
    for f in files():
        p = OUT / f
        a, sr = librosa.load(p, sr=16000, mono=True)
        f0 = librosa.yin(a, fmin=70, fmax=400, sr=sr, frame_length=1024)
        rms = librosa.feature.rms(y=a, frame_length=1024, hop_length=256)[0][: len(f0)]
        voiced = f0[(rms > 0.3 * np.median(rms)) & (f0 < 390) & (f0 > 75)]
        res[f] = dict(seconds=round(len(a) / sr, 2), f0_median=round(float(np.median(voiced)), 1) if voiced.size else None,
                      f0_iqr=round(float(np.subtract(*np.percentile(voiced, [75, 25]))), 1) if voiced.size else None)
    (OUT / "pitch.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    hi = sorted((v["f0_median"] or 0, k) for k, v in res.items())
    for f0, k in hi[-40:]:
        print(f0, k, res[k]["seconds"])


if __name__ == "__main__":
    low_priority()
    {"fetch": fetch, "analyse": analyse}[sys.argv[1]]()
