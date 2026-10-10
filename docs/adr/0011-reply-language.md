# 0011 — Danish in, English out is acceptable

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
Danish speech recognition is solved well enough locally (edda-v0.2: 7.5 % WER at ~240 ms). Danish
speech synthesis is not: the good-sounding voices (Røst-v3 Chatterbox) take 2–3 s per sentence,
and the fast one (Piper) sounds robotic. English voices are both fast and good (Pocket TTS ~90 ms
to first audio on CPU, Kokoro ~250 ms on CUDA). Mikkel is fine with Grace answering in English as
long as he can speak Danish to her.

## Decision
- **Input:** Danish (and English) speech. STT stays Danish-tuned.
- **Output:** Grace replies in English by default, with an English voice chosen by blind
  listening test (Pocket TTS vs Kokoro). The LLM is prompted to understand Danish and answer in
  English, keeping Danish names and places as they are.
- **Later:** a Danish voice returns when one is fast enough (streaming Chatterbox, an ONNX port,
  or a new model). The voice is an adapter, so this is a config change, not a redesign.

## Consequences
- Removes the slowest stage from the latency budget.
- Danish names in English sentences must sound acceptable: include them in the listening test.
- STT must keep coping with English words and code-switching in Danish speech.
