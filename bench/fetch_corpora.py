"""Downloads the public corpora grace-sim builds its test scenes from (all gitignored).

  bench/.venv/Scripts/python.exe bench/fetch_corpora.py coral   # CoRal v3 conversation test (gated: accept terms on HF)
  bench/.venv/Scripts/python.exe bench/fetch_corpora.py musan   # MUSAN noise/music/speech, CC BY 4.0 (openslr 17, ~11 GB, slow mirror)
  bench/.venv/Scripts/python.exe bench/fetch_corpora.py demand  # DEMAND room noise, CC BY 4.0 (one channel per environment kept)
  bench/.venv/Scripts/python.exe bench/fetch_corpora.py esc50   # ESC-50 sound events (breathing, typing, cough...), CC BY-NC 3.0

CoRal (OpenRAIL-D) may be used for recognition and turn detection, never for speech synthesis.
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "fixtures"
CORPORA = ROOT / "corpora"


def coral():
    from huggingface_hub import hf_hub_download

    for i in range(6):
        name = f"conversation/test-{i:05d}-of-00006.parquet"
        print(hf_hub_download("CoRal-project/coral-v3", name, repo_type="dataset",
                              local_dir=ROOT / "coral-v3"))


def musan():
    tgz = CORPORA / "musan.tar.gz"
    out = CORPORA / "musan"
    if (out / "noise").exists():
        print("musan already extracted")
        return
    CORPORA.mkdir(parents=True, exist_ok=True)
    if not tgz.exists():
        url = "https://openslr.trmal.net/resources/17/musan.tar.gz"
        print("downloading", url)
        urllib.request.urlretrieve(url, tgz.with_suffix(".part"))
        tgz.with_suffix(".part").rename(tgz)
    with tarfile.open(tgz) as t:
        t.extractall(CORPORA, filter="data")
    tgz.unlink()
    print("musan ready:", out)


def demand():
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    out = CORPORA / "demand"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(22):
        name = f"data/train-{i:05d}-of-00022.parquet"
        p = Path(hf_hub_download("voice-biomarkers/DEMAND-acoustic-noise", name,
                                 repo_type="dataset", local_dir=out / "parquet"))
        t = pq.read_table(p)
        for audio, ch in zip(t.column("audio").to_pylist(), t.column("file_name").to_pylist()):
            if ch == "ch01":
                env = Path(audio["path"] or f"env{i:02d}").parts[0].split(".")[0] or f"env{i:02d}"
                a, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                sf.write(out / f"{i:02d}-{env}.wav", a, rate)
                print("demand", i, env, rate)
        p.unlink()


def esc50():
    import zipfile

    out = CORPORA / "esc50"
    if out.exists():
        print("esc50 already there")
        return
    CORPORA.mkdir(parents=True, exist_ok=True)
    z = CORPORA / "esc50.zip"
    urllib.request.urlretrieve("https://github.com/karoldvl/ESC-50/archive/master.zip", z)
    with zipfile.ZipFile(z) as f:
        f.extractall(CORPORA)
    (CORPORA / "ESC-50-master").rename(out)
    z.unlink()
    print("esc50 ready:", out)


if __name__ == "__main__":
    for what in sys.argv[1:]:
        {"coral": coral, "musan": musan, "demand": demand, "esc50": esc50}[what]()
