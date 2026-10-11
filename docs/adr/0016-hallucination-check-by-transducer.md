# 0016 — Drop invented transcripts when a second recognizer hears nothing

- **Status:** Accepted (replaces the avg_logprob filter added after the live memory sessions)
- **Date:** 2026-10-11

## Context
edda-v0.2 (Whisper) writes fluent text from non-speech. grace-sim fed it 100 clips of room
sounds (ESC-50: breathing, typing, coughing, doors, ...) and it produced text on **100/100**,
mostly the same phrases ("Tester en to tre fire fem seks syv otte.", "to" — the live headset
hallucination). Its `no_speech_prob` is 0.0 on everything (the fine-tune broke it). The
avg_logprob < -0.55 rule, calibrated on read speech, caught only 50/100 and dropped 16/40 real
distant utterances and 15/40 utterances spoken over Grace.

ekko-v1-tiny is a transducer (Parakeet family), which does not generate free text from noise: on
the same 100 clips it produced 8 single words. On CPU it takes 60-200 ms, less than edda.

| Rule (grace-sim) | non-speech caught | clean dropped | distant dropped | over Grace dropped |
|---|---|---|---|---|
| avg_logprob < -0.55 | 50/100 | 0/40 | 16/40 | 15/40 |
| ekko heard no words | 92/100 | 0/40 | 4/40 | 1/40 |

## Decision
The speech server runs ekko on the CPU in parallel with edda and returns its text as
`check_text`. The engine drops edda's transcript when ekko heard no words. edda stays the
transcriber (14 % vs 23 % WER on clean CoRal).

## Consequences
- No added latency (ekko finishes before edda); one more model in RAM (~150 MB).
- The VAD remains the first line: room sounds triggered it once in 100 scenes.
- ekko's licence is a custom composite (NVIDIA Open Model + CoRal + CC-BY): fine for this private
  project; re-check before any distribution.
- Next: Calm-Whisper-style fine-tune of edda on non-speech → empty (data now exists in grace-sim).
