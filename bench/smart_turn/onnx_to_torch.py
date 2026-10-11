"""Recover PyTorch weights for Smart Turn v3.2 from the published fp32 ONNX (only ONNX is published).

Named initializers map 1:1 to SmartTurnV3Model parameters (prefix "inner."). Linear weights were
exported as anonymous transposed MatMul constants (val_N); each is named by following the MatMul to
its bias Add. Whisper's k_proj has no bias: it is the remaining 384x384 MatMul in each layer.

  python bench/smart_turn/onnx_to_torch.py .runtime/models/smart-turn-v3.2-fp32.onnx <out_dir>

Verifies the rebuilt model against onnxruntime on real-looking inputs and saves a HF checkpoint.
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, "fixtures/corpora/smart-turn/code")
from train import SmartTurnV3Model  # noqa: E402  (official model class, BSD-2)
from transformers import WhisperConfig  # noqa: E402


def recover_state(onnx_path):
    m = onnx.load(onnx_path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    consumers = defaultdict(list)
    for n in m.graph.node:
        for i in n.input:
            consumers[i].append(n)
    state = {}
    for name, a in inits.items():
        if name.startswith("inner."):
            state[name[len("inner."):]] = torch.from_numpy(a.copy())
    unnamed = {}
    for n in m.graph.node:
        if n.op_type != "MatMul":
            continue
        w = [i for i in n.input if i in inits and not i.startswith("inner.")]
        if not w:
            continue
        w = w[0]
        bias = None
        for c in consumers[n.output[0]]:
            if c.op_type == "Add":
                b = [i for i in c.input if i.startswith("inner.") and i.endswith(".bias")]
                if b:
                    bias = b[0]
        unnamed[w] = (n.name, bias)
    k_candidates = defaultdict(list)
    for w, (node, bias) in unnamed.items():
        arr = inits[w]
        if bias is not None:
            state[bias[len("inner."):-len(".bias")] + ".weight"] = torch.from_numpy(arr.T.copy())
        else:
            k_candidates[w].append(node)
    # remaining: per-layer k_proj (no bias) and pool_attention.2 (bias is a scalar Add we missed)
    leftovers = [w for w, (_, b) in unnamed.items() if b is None]
    return state, leftovers, inits, m


def main():
    src, out = sys.argv[1], sys.argv[2]
    state, leftovers, inits, m = recover_state(src)
    cfg = WhisperConfig.from_pretrained("openai/whisper-tiny")
    model = SmartTurnV3Model(cfg).eval()
    want = model.state_dict()
    missing = [k for k in want if k not in state]
    print("missing before k/pool fix:", missing)
    print("leftover MatMul consts:", [(w, inits[w].shape) for w in leftovers])
    # Assign leftovers by graph order: MatMul nodes appear in layer order; k_proj ones are 384x384.
    order = [n for n in m.graph.node if n.op_type == "MatMul" and any(i in leftovers for i in n.input)]
    k_names = sorted([k for k in missing if k.endswith("k_proj.weight")],
                     key=lambda s: int(re.search(r"layers\.(\d+)", s).group(1)))
    other = [k for k in missing if not k.endswith("k_proj.weight")]
    ws = [[i for i in n.input if i in leftovers][0] for n in order]
    k_ws = [w for w in ws if inits[w].shape == (384, 384)]
    o_ws = [w for w in ws if inits[w].shape != (384, 384)]
    assert len(k_ws) == len(k_names), (k_ws, k_names)
    for k, w in zip(k_names, k_ws):
        state[k] = torch.from_numpy(inits[w].T.copy())
    for k in other:
        if k.endswith(".weight"):
            cand = [w for w in o_ws if inits[w].T.shape == tuple(want[k].shape)]
            assert len(cand) == 1, (k, cand)
            state[k] = torch.from_numpy(inits[cand[0]].T.copy())
        else:  # scalar bias (pool_attention.2.bias, classifier.6.bias)
            scal = [n for n in m.graph.initializer if numpy_helper.to_array(n).size == 1]
            print("scalar bias needed for", k, "candidates", [(s.name, numpy_helper.to_array(s)) for s in scal])
    still = [k for k in want if k not in state]
    print("still missing:", still)
    return state, still, model, src, out




def verify_and_save():
    import glob
    import onnxruntime as ort
    import soundfile as sf
    sys.path.insert(0, os.path.dirname(__file__))
    import common
    state, still, model, src, out = main()
    assert not still
    model.load_state_dict(state, strict=True)
    model.eval()
    sess = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    clips = sorted(glob.glob("bench-results/turn/*.wav"))[:32]
    x = np.stack([common.features(sf.read(f, dtype="float32")[0]) for f in clips]) if clips \
        else np.random.randn(8, 80, 800).astype(np.float32)
    ref = sess.run(None, {"input_features": x})[0][:, 0]
    with torch.no_grad():
        got = model(torch.from_numpy(x))["logits"].numpy()[:, 0]
    d = np.abs(ref - got).max()
    print(f"torch vs onnx on {len(x)} clips: max |dp| = {d:.2e}")
    assert d < 1e-4
    model.save_pretrained(out)
    print("saved", out)


if __name__ == "__main__":
    verify_and_save()
