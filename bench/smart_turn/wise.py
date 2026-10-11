"""WiSE-FT: interpolate the fine-tuned weights with the v3.2 base (theta = (1-a)*base + a*ft) to
trade Danish gains against multilingual forgetting, and export each mix as ONNX fp32.

  python bench/smart_turn/wise.py bench-results/smart_turn/runs/da-v1/last 0.5 0.7
"""
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, "fixtures/corpora/smart-turn/code")
from train import SmartTurnV3Model, export_to_onnx_fp32, CONFIG  # noqa: E402

base = SmartTurnV3Model.from_pretrained("bench-results/smart_turn/base_torch").eval()
ft_dir = sys.argv[1]
ft = SmartTurnV3Model.from_pretrained(ft_dir).eval()
sb, sf = base.state_dict(), ft.state_dict()
for a in map(float, sys.argv[2:]):
    m = SmartTurnV3Model.from_pretrained("bench-results/smart_turn/base_torch").eval()
    m.load_state_dict({k: (1 - a) * sb[k] + a * sf[k] if sb[k].is_floating_point() else sf[k] for k in sb})
    out = f"{os.path.dirname(ft_dir)}/wise{int(a * 100)}"
    os.makedirs(out, exist_ok=True)
    m.save_pretrained(out)
    print(export_to_onnx_fp32(m, f"{out}/model_fp32.onnx", CONFIG))
