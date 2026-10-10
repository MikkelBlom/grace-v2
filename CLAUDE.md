# Grace v2

A 100 % local, Danish-first voice assistant for Mikkel's Windows laptop (RTX 5090 Laptop 24 GB,
64 GB RAM, Core Ultra 9 with NPU + Arc iGPU). Rust core, models behind adapters, every model
chosen by bake-off. Start with [README.md](README.md), then `ai-instructions/INDEX.md`.

## Must never be violated
- **Local only.** No cloud AI. The internet is for tools like web search, never for inference.
- **Latency is the product.** Anything on the voice path must stream, be cancellable, and be
  traced. Target: < 800 ms from speech end to first audio on simple turns.
- **Never block the audio thread.** No locks, allocation-heavy work or IO on it; tracing is
  non-blocking by design.
- **Grace only answers people.** Her own voice and anything the PC plays must never trigger her
  ([ADR 0006](docs/adr/0006-echo-cancellation.md)). The media false-trigger test stays at 0.
- **Live test before done.** A milestone is not done until it was used with the speakers on and
  its trace was checked.
- **Measure, don't guess.** Model and design choices are backed by `grace-bench` results or
  traces, recorded in `docs/MODELS.md` or an ADR.
- **Small scope.** Only the tools a milestone needs. New ideas go to `ai-instructions/FUTURE_IDEAS.md`
  or a GitHub issue, not into the current change.
- **No secrets, personal recordings or traces in git.** Fixture audio and traces are gitignored.
- **The model can't approve itself.** Risky tool actions need a fresh yes from a person.

## Conventions
- Pure logic in `grace-core` (inputs + explicit clock → commands), unit-tested without IO.
- Decisions: add an ADR in `docs/adr/` (short; supersede, don't rewrite).
- Stage span names (`stage = "stt"`, `"llm.ttft"`, …) are a contract with `grace-debug`.
- Work tracking: GitHub issues + milestones on `MikkelBlom/grace-v2` (and the project board).
- Grace v1 (reference only): `C:\Users\mikke\Documents\Claude\Projects\AI automation\grace`.
