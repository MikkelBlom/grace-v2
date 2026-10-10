# 0005 — VAD + turn-end model + speculative STT

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
Target: near-instant replies (aim < 800 ms speech end → first audio on simple turns). v1 used
silence-only endpointing, which either cut Mikkel off or waited too long, and generated a whole
JSON object before speaking anything.

## Decision
- **VAD:** Silero VAD v6 gates audio frames.
- **End of turn:** Pipecat Smart Turn v3.2 (ONNX, CPU, supports Danish) judges each pause; a
  silence timeout is the fallback when it is unsure.
- **Speculation:** STT starts the moment speech pauses, in parallel with the turn-end model. If
  the user resumes, the result is discarded. LLM prefill can start speculatively too.
- **Streaming out:** the LLM streams; the reply is cut into sentences and each sentence goes to
  TTS as soon as it is complete. First audio = first sentence.
- **Interruptions:** user speech during a reply first *ducks* Grace's audio; only speech that
  lasts `min_barge_in` (default 350 ms) cancels the reply. Cancellation is a per-turn token
  that stops LLM generation, TTS synthesis and playback together.
- The policy is the pure state machine `grace_core::turn::TurnMachine`.

## Consequences
- Wasted work on false pauses (discarded STT/LLM prefill). Cheap compared with waiting.
- Barge-in depends on echo cancellation working ([0006](0006-echo-cancellation.md)).
- Thresholds (`turn_end_threshold`, `max_trailing_silence`, `min_barge_in`) must be tuned from
  real traces, not guessed.
