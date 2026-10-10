# 0006 — Cancel all PC audio via a loopback reference

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
Mikkel mostly uses speakers, not a headset. Grace must not respond to herself, YouTube or
Spotify, but other people in the room should be able to talk to her. Voice verification
(respond only to Mikkel) would block the second requirement.

## Decision
Acoustic echo cancellation with **WASAPI loopback of the render endpoint as the far-end
reference**. The loopback carries everything the speakers play (Grace, media, calls), so all of
it is subtracted from the microphone signal; people in the room are never in the loopback.

Candidates, decided by bake-off on this laptop:
- **A.** Windows communications-mode AEC with `IAcousticEchoCancellationControl::SetEchoCancellationRenderEndpoint` (OEM/Voice Clarity APO, zero DSP code; may be unavailable on this driver).
- **B.** Loopback → WebRTC AEC3 (`webrtc-audio-processing` or the pure-Rust `aec3`) → DeepFilterNet residual suppression.
- **C.** A + DeepFilterNet.

Metrics: echo return loss enhancement, false triggers during 30 min of media playback (must be
0), barge-in recall from Mikkel and from a second person.

Backstops: reject transcripts that closely match what Grace is currently saying; raise the VAD
threshold while she speaks.

No voice verification for now. It can be added later as an optional layer (sherpa-onnx speaker
embeddings) without changing this design.

## Consequences
- Clock drift between a USB mic and the speakers needs compensation; the built-in mic usually
  shares the codec clock.
- Reference implementation worth reading first: KoljaB/echoff (WASAPI loopback + WebRTC APM).

## Update 2026-10-10
`crates/grace-audio/examples/aec_probe.rs`: all three microphones on this laptop (built-in Realtek
array, USB mic, webcam mic) expose `IAcousticEchoCancellationControl` and accept the speakers as
reference; AEC, noise suppression and beamforming are on by default in Communications mode.
Option A is therefore available. Next: measure echo reduction with
`aec_test --play` (plays a clip, records raw and AEC'd mic in parallel).
