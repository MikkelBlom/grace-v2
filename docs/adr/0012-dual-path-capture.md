# 0012 — Two capture paths: one decides *when*, one hears *what*

- **Status:** Accepted
- **Date:** 2026-10-11

## Context
Measured on Mikkel's setup (Logitech Z337 over Bluetooth, laptop mic array; double-talk run with
him reading five Danish lines while Grace's English voice played):

| Signal | VAD false speech while only Grace talks | His 5 lines, WER (edda) | Grace's words in transcript |
|---|---|---|---|
| Raw mic | ~15.7 s of 17 s | 18 % | 0 |
| Windows communications mode (AEC/NS/beamforming) | ~15.2 s | **9 %** | 0 |
| Our AEC3 + NS (adapted, talking boost) | **0.0 s** | 32–35 % | 0 |
| Our AEC3 linear output (pre-suppression) | ~13 s | 68 % | 0 |

Our AEC3 is the only signal on which Grace's voice and YouTube don't trigger VAD, but its
residual suppressor also damages near-end speech during double-talk. Windows' processing leaves
the echo (useless for VAD) but keeps Mikkel's voice cleanest. The Danish-forced STT model does
not transcribe Grace's English speech, so echo in the STT signal is harmless while she answers
in English ([0011](0011-reply-language.md)).

## Decision
Open the microphone twice:
- **Raw stream → our AEC3 + NS → VAD → turn machine.** Decides when someone is speaking and when
  Grace is interrupted.
- **Communications-mode stream (Windows voice processing) → STT.** Transcribes only the spans the
  first path marked as speech (plus padding), so media alone is never transcribed.

## Consequences
- Two capture streams from one device; both are timestamped, so spans map between them exactly.
- If Grace answers in Danish later, her echo would reach the STT signal: revisit then (better AEC
  tuning or transcript-vs-own-speech rejection).
- Danish speech from media during a real user turn can leak into that turn's transcript.
- AEC3 suppressor tuning for double-talk remains worth doing; if it gets good enough, the paths
  can merge.
