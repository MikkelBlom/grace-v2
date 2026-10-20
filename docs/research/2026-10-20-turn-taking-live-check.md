# Live check: merging continuations and prepared replies (ADR 0018)

Setup: DJI Mic 2 clip mic + JBL Tune 777NC (stereo "Hovedtelefoner"). Check the startup line
names the DJI as the mic. `./scripts/grace.ps1`, then afterwards
`python bench/trace_turns.py traces/engine-<id>.jsonl`.

A short session (10-15 minutes) is enough; this is a final check, not a test set.

1. **Simple questions** ("hvad er klokken", "hvordan går det"): Grace should answer about as
   fast as before or faster, never slower. Trace: `prepared reply released` on most turns.
2. **Pause mid-thought and go on** ("jeg tænker på ... [1 s] ... om jeg skal tage toget"):
   Grace should not answer the first half. If she starts, she should stop within a word or two
   and then answer the whole sentence. Terminal: `(listening on)` then one `you:` line with the
   whole sentence; trace: `turn reopened`, `reply retracted`.
3. **Finish, then add something right after she starts** (within ~1 s of her first word):
   same as 2: her first words are dropped and she answers both parts together.
4. **Interrupt later** (a few seconds into a long answer, "vent, stop"): she stops and answers
   the interruption. Afterwards ask "hvad sagde du lige?": she should only refer to what she
   actually said before you cut in (history holds only what played).
5. **Cough or say "hm" right as she starts**: she should carry on (`(carries on)`), not
   restart or answer the cough.
6. **Long pause on a filler** ("og øh ... [2 s]"): she should keep waiting, not jump in.
7. **Silence after a finished sentence that Smart Turn doubts**: she should answer after the
   silence fallback, with her first audio right at that moment (the reply was prepared).

Things to report back: any time she answered half a sentence and did *not* take it back, any
time she went silent and never answered, any click or crackle when she stopped, and whether
the stop-and-restart in 2-3 feels acceptable or annoying.
