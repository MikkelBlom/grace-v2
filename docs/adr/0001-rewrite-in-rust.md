# 0001 — Rebuild as Grace v2 with a Rust core

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
Grace v1 (TypeScript/Node, Electron, 4–5 Python model servers, ~100 tools) never became usable
day to day. Turns took 17–40 s, launch took 2–3 minutes, and most features were never tested
live. The causes were architectural (see [LESSONS-FROM-V1](../LESSONS-FROM-V1.md)): a
non-streaming JSON loop, no cancellation model, shared mutable state between conversation and
missions, and breadth before depth. Retrofitting streaming and cancellation onto v1 would be
duct tape.

Grace runs while Mikkel develops: dev servers, databases and VMs compete for the same machine.

## Decision
Start a new repository, Grace v2, with the core in Rust. Carry over the *lessons* from v1, not
its code. v1 stays where it is as read-only reference.

Rust is chosen for:
- **Resource use.** One native process instead of Node + Electron + several Python processes
  that each load PyTorch (1–2 GB RAM apiece). Near-zero idle CPU.
- **Real-time audio.** No GC pauses on the audio thread; echo cancellation ([0006](0006-echo-cancellation.md)) needs a tight loop.
- **Correctness under AI-driven development.** The compiler and type system catch whole classes
  of mistakes before they reach a live test.

Rust is *not* expected to make model inference faster. Inference runs in C++/CUDA kernels
(llama.cpp, whisper.cpp/CTranslate2, ONNX Runtime) whatever language calls them. Latency wins
come from streaming, speculation and caching ([0005](0005-turn-taking.md)).

## Consequences
- Compile times work against "change takes effect immediately". [0002](0002-process-split-and-hot-reload.md) compensates.
- Some models are PyTorch-only today (Chatterbox). They run as sidecar processes until an ONNX
  path exists; the engine treats every model as a replaceable adapter.
- Fewer ready-made voice-agent frameworks in Rust; we assemble from `sherpa-onnx`, `ort`,
  `whisper-rs`, `wasapi`, llama.cpp.
