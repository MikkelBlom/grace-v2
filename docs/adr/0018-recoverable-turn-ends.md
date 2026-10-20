# 0018 — Recoverable turn ends: merge continuations, prepare replies early

- **Status:** Proposed (live check pending)
- **Date:** 2026-10-20
- Supersedes the turn-end thresholds and "LLM prefill can start speculatively" of
  [0005](0005-turn-taking.md); the rest of 0005 stands.

## Context
Live run 4: speech end → first audio p50 1.38 s, p90 3.66 s; 11/48 turns waited the ~3 s long
timeout because Smart Turn da1 scored a finished sentence low. The turn-end policy was cautious
because jumping in early was expensive: if Mikkel went on talking, the second half became a
new turn answering Grace's half-answer, and history kept every sentence TTS had *synthesized*,
played or not. Mikkel: never answering is worse than jumping in early.

`grace-turn-sim` (new) drives the real `TurnMachine` in virtual time over 250 CoRal turns
(24 unseen speakers; 150 resumed after a 0.35-3 s pause, 100 single), with the real frontend,
Smart Turn da1 and edda STT, and the LLM + TTS path at the live-run-4 median (620 ms). Its
baseline reproduces live run 4: single turns p50 1.44 s / p90 3.6 s (live 1.38 / 3.66).

## Decision
1. **Merge continuations.** Speech that starts before Grace's reply is audible (at once: no
   echo is possible yet) or within 1.5 s after it became audible (after `min_barge_in`) reopens
   the *same* turn. When it ends, the whole turn's audio is transcribed again (WER 16.4 % vs
   18.7 % for joining the two transcripts, 4,238 words). Same words → the paused reply resumes
   (it was a cough); new words → the reply is retracted (never enters history or memory) and
   the whole turn is answered. Later speech is an interruption as before.
2. **History holds what was heard.** The render thread counts played samples per reply; an
   interrupted reply enters history and memory as the sentences that played, the cut one
   shortened to its share of words and ending in "—". Memory writes moved to the moment a
   reply is final.
3. **Prepared replies.** A pause with a transcript that isn't committed yet starts the reply
   with its audio held (first sentence only); the commit releases it if the conversation is
   unchanged. Resuming discards it.
4. **Bolder policy.** Silence fallback 1.6 → 1.0 s, long wait (score < 0.05 or a hanging word)
   3.0 → 2.0 s, VAD hangover 256 → 128 ms. Threshold stays 0.3 (0.2 and 0.5 changed little).
   `min_commit_silence` exists but stays 0: at 500-700 ms it changed nothing, since commits
   already come > 600 ms into a pause.

## Consequences
- Sim, before → after: paused turns answered in halves 63 % → 3 % (heard half-answers
  44 % → 2 %); p90 speech end → first audio 3.6-3.7 → 2.0-2.1 s; p50 1.44 → 1.24 s on single
  turns. Ablation (new thresholds without 1-3): 75 % halves, p90 2.6 s.
- The visible cost: in 43 % of paused turns Grace starts (0.4 s on average) and stops when
  Mikkel goes on. Whether that feels fine is the live check's question.
- More work per turn: 2.8 vs 1.9 LLM requests per paused turn (0.7 discarded prepares), on
  llama-server's 2 slots shared with memory learning; more STT calls from the shorter hangover.
- p50 is now bounded by the pipeline, not the policy: STT (~300 ms live, ~450 in the sim) +
  recall (~90) + TTFT (~200) + Pocket TTS first audio (~500 live vs 94 bench). < 800 ms needs
  those, not more turn-taking.
- The shorter hangover also ends overlap sooner while Grace talks: a barge-in with gaps over
  128 ms may need a second try. Watch it live.
