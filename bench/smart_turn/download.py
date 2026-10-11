"""Downloads the Smart Turn v3.2 datasets (CC BY 4.0) and CoRal v3 (OpenRAIL-D, gated) shards.

  bench/.venv-smartturn/Scripts/python bench/smart_turn/download.py [what...]
what: st-test st-train coral-conv coral-read  (default: all). Train sets are shard subsets.
"""
import sys
from huggingface_hub import snapshot_download

ROOT = "fixtures/corpora/smart-turn"
JOBS = {
    "st-test": ("pipecat-ai/smart-turn-data-v3.2-test", f"{ROOT}/v3.2-test", None),
    # Replay subset of the multilingual train set (every 3rd of 83 shards, ~14 GB, ~90k clips):
    # enough to keep the other languages from being forgotten during a short fine-tune.
    "st-train": ("pipecat-ai/smart-turn-data-v3.2-train", f"{ROOT}/v3.2-train",
                 [f"data/train-{i:05d}-*" for i in range(0, 83, 3)] + ["README.md", "LICENSE"]),
    "coral-conv": ("CoRal-project/coral-v3", "fixtures/coral-v3",
                   ["conversation/*", "README.md", "LICENSE"]),
    "coral-read": ("CoRal-project/coral-v3", "fixtures/coral-v3",
                   [f"read_aloud/train-{i:05d}-*" for i in range(0, 313, 39)]
                   + ["README.md", "LICENSE"]),
}
for k in sys.argv[1:] or JOBS:
    repo, dst, pat = JOBS[k]
    print("downloading", k, flush=True)
    snapshot_download(repo, repo_type="dataset", local_dir=dst, allow_patterns=pat, max_workers=8)
    print("done", k, flush=True)
