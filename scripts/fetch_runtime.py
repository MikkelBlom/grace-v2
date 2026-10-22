"""Downloads the pinned LLM runtime and model into .runtime/ (gitignored).

  python scripts/fetch_runtime.py

Pins live here so every machine runs the same build. Bump deliberately, re-run the latency
bench, and record the result in docs/MODELS.md.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent / ".runtime"
LLAMA_BUILD = "b11539"
LLAMA_ASSETS = [
    f"llama-{LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
]
FILES = [
    # (url, name): plain downloads, pinned by tag.
    ("https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx", "silero_vad_v6.2.3.onnx"),
]
MODELS = [
    # (repo, file): Google's quantization-aware-trained Q4_0, better than post-hoc Q4 at the same size.
    ("google/gemma-4-12B-it-qat-q4_0-gguf", "gemma-4-12b-it-qat-q4_0.gguf"),
    # Gemma's vision projector from the same repo: look_at_screen, read_image (docs/tools/vision.md).
    ("google/gemma-4-12B-it-qat-q4_0-gguf", "mmproj-gemma-4-12b-it-qat-q4_0.gguf"),
    # Smart Turn v3.2 full precision: the int8 "-cpu" build swings ±0.3 near the decision
    # threshold depending on ONNX Runtime settings (2026-10-11).
    ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx"),
]


def fetch_llama():
    dest = ROOT / "llama.cpp" / LLAMA_BUILD
    if (dest / "llama-server.exe").exists():
        print(f"llama.cpp {LLAMA_BUILD}: present")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for asset in LLAMA_ASSETS:
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{asset}"
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    print(f"llama.cpp {LLAMA_BUILD}: {dest}")


def fetch_files():
    dest = ROOT / "models"
    dest.mkdir(parents=True, exist_ok=True)
    for url, name in FILES:
        if not (dest / name).exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest / name)
        print(f"file: {dest / name}")


def fetch_models():
    for repo, name in MODELS:
        path = Path(hf_hub_download(repo, name, local_dir=ROOT / "models"))
        if name == "smart-turn-v3.2-gpu.onnx":
            # The "gpu" file is the fp32 model; it runs on CPU too. Give it an honest name.
            target = path.with_name("smart-turn-v3.2-fp32.onnx")
            if not target.exists():
                target.write_bytes(path.read_bytes())
            path = target
        print(f"model: {path}")


if __name__ == "__main__":
    fetch_llama()
    fetch_files()
    fetch_models()
