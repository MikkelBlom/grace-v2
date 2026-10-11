# 0013 — One capture path: STT reads the echo-cancelled signal

- **Status:** Accepted (supersedes the STT half of [0012](0012-dual-path-capture.md))
- **Date:** 2026-10-11

## Context
ADR 0012 transcribed from a second mic stream opened in Windows communications mode (Windows'
own voice processing, with the speakers set as echo reference), because on the Z337 speakers it
gave the best transcripts during double-talk (9 % WER vs 32 % on our AEC3 path). Live use then
showed:
- With the Bluetooth headset as output, that stream returned near-silence: all 9 turns
  transcribed from it came back as the hallucination "to"; all 9 turns transcribed from our
  echo-cancelled path were correct.
- The Windows Audio service hung twice (03:28 and 04:51 on 2026-10-11, same crash bucket), each
  time during a session that had a communications-mode stream with an echo reference set on a
  Bluetooth device. Grace went deaf when it hung. This is circumstantial, but strong.

## Decision
Capture only the raw mic and the speaker loopback. VAD, Smart Turn and STT all read our AEC3
output. Grace never opens a communications-mode stream or sets an echo reference (the
`aec_probe`/`aec_test` tools still can, for diagnostics only).

## Consequences
- Simpler engine: two capture streams instead of three, less CPU.
- Transcripts during double-talk (talking over Grace) are worse than ADR 0012 measured; tune
  AEC3's suppressor if interruptions get misheard.
- Audio threads now reopen their devices after errors and every thread is watched, so a device
  failure is visible and recoverable instead of silent.
