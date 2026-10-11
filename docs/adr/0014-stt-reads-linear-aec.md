# 0014 — STT reads AEC3's linear output; test on simulated rooms, not on Mikkel

- **Status:** Accepted (refines [0013](0013-single-capture-path.md))
- **Date:** 2026-10-11

## Context
Live misrecognitions could not be diagnosed by more live sessions: Mikkel is not the test set.
`grace-sim` now runs the engine's real frontend (AEC3 → VAD → gate) faster than real time on
scenes built from CoRal v3 conversation speech, simulated rooms (pyroomacoustics), real noise,
and Grace's own TTS voice, with echo levels calibrated to real Z337 recordings. Research
(docs/research/2026-10-11-foundations.md) says nonlinear suppression artifacts hurt ASR more
than leftover noise.

Pilot (40 scenes per set, edda int8, cuts at the true speech edges):

| WER | raw mic | AEC3 linear | AEC3 full chain (was STT input) |
|---|---|---|---|
| clean CoRal (baseline) | 14.3 % | — | — |
| person 1.5-3 m away | 43.8 % | 46.0 % | 48.8 % |
| person talks over Grace | 41.3 % | 40.3 % | 47.4 % |

Full sets (150 utterances each, real DEMAND room noise, 2026-10-11):

| WER | raw mic | AEC3 linear | AEC3 full chain |
|---|---|---|---|
| near (0.4-0.8 m) | 20.8 % | 21.9 % | 22.2 % |
| far (1.5-3 m) | 26.5 % | 27.2 % | 28.0 % |
| person talks over Grace | 36.6 % | 32.1 % | 43.5 % |

## Decision
STT transcribes AEC3's linear-filter output (16 kHz, before residual-echo and noise
suppression). VAD and Smart Turn keep the full chain. Turn audio saves all three signals.
Speech-pipeline changes are measured with `grace-sim` + `grace-bench stt` on public corpora.

## Consequences
- 5-7 WER points better than the full chain in the pilot, at no latency cost.
- Raw mic is as good when Grace is silent, but carries her echo when she isn't; linear is the
  one signal that is good in both.
- Confirmed on the full sets: linear is ~1 point behind raw when Grace is silent and 4.5 points
  ahead of raw (11 ahead of the full chain) when she talks.
- Decoding: fp16 beam 5 at temperature 0 is 0.3-2 points better (near 21.4, far 26.9, double-talk
  30.1 %); adopt only after its latency is measured on an idle GPU.
