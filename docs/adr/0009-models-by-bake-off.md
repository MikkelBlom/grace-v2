# 0009 — Choose models by bake-off on this machine

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
The model landscape changes monthly, leaderboards are measured on other hardware and other
audio, and Danish quality claims are mostly self-reported. v1 picked models from research and
then discovered problems live.

## Decision
- Every model slot (VAD, turn-end, STT, LLM, TTS, embedder, reranker, AEC) is an adapter behind
  a trait. Swapping a model is a config change.
- Candidates are compared with `grace-bench` on our own fixtures, on this laptop, measuring
  accuracy and warm latency on the runtime we would ship. Results accumulate in
  `bench-results/summary.jsonl`; the current picks and numbers live in [MODELS.md](../MODELS.md).
- A pick is re-opened when a new candidate appears (a periodic "what's new" scan is planned).
- Start small and fast (Gemma 4 E4B), and only move up when traces show the harness is the
  limit, not the model.

## Consequences
- Bench runners can be any language (JSON-lines protocol), so PyTorch-only candidates can be
  measured before anyone ports them.
- Python runners measure Python-runtime latency; production latency is confirmed again once
  the Rust adapter exists.
