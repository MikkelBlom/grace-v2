# 0004 — Pure core, recorded fixtures, measured budgets

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
v1 shipped features as "headless-verified, needs live test", and the live tests rarely
happened. Voice behaviour (turn-taking, interruptions, echo) is timing-dependent and hard to
test by hand.

## Decision
Five layers of tests:

1. **Pure logic** (`grace-core`): state machines take inputs plus an explicit clock and return
   commands. Deterministic unit tests, no audio, no threads. Turn-taking lives here.
2. **Adapters behind traits.** Every model (VAD, turn-end, STT, LLM, TTS, embedder) sits behind
   a trait with a fake implementation, so the whole pipeline runs in CI without a GPU.
3. **Recorded audio fixtures.** Danish speech, speech over YouTube/Spotify playing through the
   speakers, and Grace's own voice. Replayed through the real pipeline. False-trigger rate on
   media playback is a test that must stay at **0**.
4. **Bake-offs** (`grace-bench`): candidate models measured on this machine for accuracy and
   latency, results appended to `bench-results/summary.jsonl` ([0009](0009-models-by-bake-off.md)).
5. **Latency budgets.** Per-stage budgets checked from traces; a regression fails the run.

A milestone is done when it has been used live, with the speakers on, and its trace looks right.

## Consequences
- CI runs layers 1–2 on every push. Layers 3–5 need the GPU machine and run locally.
- Fixture audio is never committed (dataset licences, privacy). Scripts recreate it.
