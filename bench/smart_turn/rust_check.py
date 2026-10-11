"""Compare turn_any (Rust, grace-vad) probabilities with Python onnxruntime on the same WAVs.

  python bench/smart_turn/rust_check.py <model.onnx> <turn_any output.tsv>
"""
import sys; sys.path.insert(0,'bench/smart_turn'); import common, numpy as np, soundfile as sf, onnxruntime as ort
model, tsv = sys.argv[1], sys.argv[2]
r={l.split('\t')[0]:float(l.split('\t')[1]) for l in open(tsv)}
s=ort.InferenceSession(model,providers=['CPUExecutionProvider'])
d=[];ok=0
for f,p in r.items():
    x=common.features(sf.read(f,dtype='float32')[0])[None]; q=float(s.run(None,{'input_features':x})[0][0][0]); d.append(abs(p-q)); ok+=(p>=0.5)==f.endswith('_complete.wav')
print(len(r),'clips; rust acc',ok/len(r),'max |rust-python|',max(d))
