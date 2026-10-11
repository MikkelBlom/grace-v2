# 0015 — Memory: facts rendered by code, a commitments table, quoted evidence

- **Status:** Accepted (revises [0007](0007-memory-model.md))
- **Date:** 2026-10-11

## Context
Live sessions showed memory failures that the first fixes only patched (docs/research/
2026-10-11-foundations.md): a cancelled exam came back through an old free-text summary and was
re-learned from Grace's own words; a reminder sat unnoticed in a flat fact list; a correction
was dropped. Research: updates are every memory system's weak point; free-text summaries next
to facts are a second, uncorrected source of truth; agents amplify errors from their own
output; deferred intentions need a typed store run by code.

`grace-memory-bench` replays 40 hand-written Danish multi-session scenarios (fixtures/memory/
scenarios, incl. every live failure) through the real memory path with a fake clock, and checks
the stored state and Grace's answers. Baseline (v0.2 code): 206/246 checks.

## Decision
- **The session block is rendered by code** from current facts and due commitments. Session
  summaries are no longer shown (kept in the DB for future episodic search).
- **Commitments are their own table** (text, earliest due date, open/done/cancelled, source
  turn). Due ones open the session block as an instruction to raise them in the first reply;
  not-yet-due ones are invisible. The extractor opens (`commit`) and closes (`done`/`cancel`)
  them by id.
- **Evidence in code:** every add/update/commit must quote Mikkel's words, and code checks the
  quote against what he said (≥ 80 % of its words). Grace's guesses can't become facts.
- **Scope:** each op says whether it is `lasting`; requests for the moment are not stored.
- **Calendar:** the extractor gets today plus the next 14 days with weekdays, and writes dates
  in words.
- **Duplicates:** embedding distance alone no longer decides; numbers, dates and names must
  match too ("15 October" vs "16 October" had been merged and a correction lost).

## Consequences
- Results per category are in docs/MODELS.md (memory section); regressions are visible per
  check in `bench-results/memory/<label>/report.json` with a trace of every memory operation.
- Still open: in-session corrections and interrupted updates depend on the per-turn extractor
  seeing the right related facts; a session-end reconciliation pass is the next candidate.
