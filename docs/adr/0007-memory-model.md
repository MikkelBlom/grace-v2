# 0007 — Remembered vs observed memory, SQLite as the source of truth

- **Status:** Accepted (design); implementation starts in v0.2
- **Date:** 2026-10-10

## Context
Memory is the second foundational feature: Grace should become a source of truth about herself,
Mikkel, how to act, and his PC. It must both evolve over time and update when facts change. v1
appended everything into a vector store and had no notion of a fact being replaced.

## Decision
Two kinds of knowledge:

- **Remembered** (only knowable because someone said it or Grace concluded it): facts about
  Mikkel, people, preferences, decisions, Grace's own identity and behaviour rules, episodes.
  Every fact carries **provenance** (source turn), **confidence**, and **bitemporal validity**
  (`valid_from`/`valid_to`, `recorded_at`/`invalidated_at`). A contradiction closes the old
  fact's validity instead of deleting it, so "what did we believe in June?" stays answerable.
  Writes go through explicit operations: ADD, UPDATE (supersede), INVALIDATE, NOOP.
- **Observed** (can be read from the machine): folders, files, installed and running programs,
  dev servers. Indexed live from the filesystem and OS, never memorised as facts, so they can't
  go stale.

Storage, all in-process:
- **SQLite** is the source of truth (facts, episodes, provenance, validity).
- **FTS5** for lexical search plus **sqlite-vec** for vectors, fused with reciprocal rank
  fusion. Tantivy (Danish stemmer) if FTS5 underperforms on Danish.
- **Embedder** (turns text into vectors; sqlite-vec only stores and searches them): chosen by
  bake-off. Candidates are `microsoft/harrier-oss-v1-270m` (MIT, best multilingual score per
  size) and `google/embeddinggemma-2` (Apache 2.0, Oct 2026, multimodal). Runs on CPU or iGPU,
  never on the RTX.
- **Consolidation** (merging episodes into facts, decay, dedup) runs in the background, off
  the voice path.

## Consequences
- Schema designed for the full model up front; v0.2 implements only facts + episodes.
- Every write must name its source, which is also the defence against memory poisoning from
  web pages or tool output.
- Changing the embedder means re-embedding; store the embedder id per vector table.
