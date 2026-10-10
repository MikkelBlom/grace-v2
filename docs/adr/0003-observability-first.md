# 0003 — Trace every stage of every turn from day one

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
In v1 it was very hard to tell where time went in a turn. Latency is the product's main
requirement, so it has to be visible all the time, not reconstructed after the fact.

## Decision
- Every process uses `grace-trace`: `tracing` spans and events written as JSONL, one file per
  process run, on a background thread behind a bounded channel. A full channel drops and counts
  records; tracing never blocks the audio thread.
- A span field `turn` tags everything inside it, so per-turn waterfalls need no bookkeeping.
- Stage spans use `stage = "<area>.<step>"` (`vad.speech`, `turn.detect`, `stt`, `llm.ttft`,
  `llm.decode`, `tool.<name>`, `tts.ttfa`, `tts.play`) plus `model = "<id>"`.
- LLM prompts, thoughts and tool calls/results are trace events too.
- `grace-debug` serves a live per-turn waterfall at `http://127.0.0.1:7700` with the headline
  number **speech end → first audio**.
- Later: each turn's audio is saved with its trace, so a bad turn can become a regression
  fixture ([0004](0004-testing-strategy.md)).

## Consequences
- Stage names are a contract between the engine and the viewer; keep them stable.
- Trace files grow; rotation and retention are needed before long-running use.
- Traces will contain transcripts. They stay local and are gitignored.
