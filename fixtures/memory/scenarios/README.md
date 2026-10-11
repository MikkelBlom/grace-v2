# Memory scenarios (Danish, multi-session)

Hand-written evaluation data for Grace's memory. A runner replays each scenario as **text** with a
**fake clock** through the real memory path (learning, session summaries/reconciliation, session
block, recall), then checks deterministic assertions. Style: LoCoMo (one persona, dated timeline)
plus LongMemEval / HaluMem categories and Grace's own live failures. Evaluation data only: never
train on it.

Validate with `bench/.venv/Scripts/python.exe bench/memory_scenarios_check.py`.

## File layout

One YAML file per scenario, `NN-short-name.yaml`; `id` equals the file stem.

```yaml
id: 01-live-stale-exam-self-echo        # == file stem
title: Short human title
categories: [knowledge_update, self_echo] # from the list below, 1-4 entries
persona: >                                # who/what this is about (always Mikkel unless noted)
  Mikkel, 24, software student in Odense ...
live_failure: >                           # optional: the live run this reproduces
  ...
sessions:                                 # 2-5, strictly increasing `start`
  - start: "2026-10-12T19:30:00+02:00"    # ISO 8601 with the Europe/Copenhagen offset in force
    turns:
      - user: "jeg har en eksamen i databaser den fjortende december"  # Danish, as STT delivers it
        grace: "Good luck with the database exam on the 14th of December."  # English reply
        gap_s: 20            # optional, seconds after the previous turn ended (default 20)
        interrupted: true    # optional: the user barged in, Grace's reply was cut off
        stt_garbled: true    # optional: this transcript is an STT error/hallucination
        stt_logprob: -0.7    # optional: avg_logprob to hand the memory system (see below)
    expect:                  # checked after the session has ended (see "Clock")
      facts_present: [...]
      facts_absent: [...]
      facts_invalidated: [...]
      commitments_open: [...]
      commitments_done: [...]
      probe_at: "2026-10-28T09:00:00+01:00"  # required on the LAST session if it has probes
      probes: [...]
```

All `expect` lists are optional; a session may have no `expect` at all.

## Categories

| id | what it tests |
|---|---|
| `extraction` | a plainly stated fact is stored (LongMemEval information extraction) |
| `multi_session` | the answer needs facts from more than one session |
| `temporal` | relative dates resolved against the fake clock; ordering; "how many days until" |
| `knowledge_update` | a later statement replaces or ends an earlier fact |
| `abstention` | never told, or only Grace said it: Grace must say she doesn't know |
| `commitment` | something Grace was asked to bring up is surfaced next session, then closed |
| `self_echo` | Grace repeats a stale/invented fact; it must not be (re-)learned from her words |
| `interrupted` | the fact is in a turn whose reply was barged in on |
| `stt_garbled` | hallucinated/garbled/background transcripts store nothing |
| `momentary_command` | "hold din kæft", "sig det igen" are not lasting preferences |
| `duplicate` | true duplicates stored once; similar but distinct facts ("two sisters") both kept |
| `dialect` | Danish dialect, slang, colloquial negation, Danish number words |
| `same_session_correction` | "… nej vent, det er fredag" in the same session: only the corrected fact survives |

## Text matching (all assertions)

A **matcher** is `match: [group, group, ...]`. A group is a string or a list of alternative
strings. A text satisfies the matcher when **every group** has at least one alternative that
matches (AND of ORs).

- Matching is case-insensitive substring after Unicode NFC normalisation; no accent folding
  (æ/ø/å alternatives are listed explicitly, e.g. `[københavn, copenhagen]`).
- An alternative starting with `re:` is a regex (Python/Rust `regex` compatible, case-insensitive,
  Unicode `\b`), e.g. `'re:\b14(th)?\b'`.
- Facts are written in English by the current extractor, so groups list English first and Danish
  as alternatives. Dates have several forms; use regex groups (`'re:\b16\b'` + `[november, nov]`).
- Optional `exclude: [...]` next to `match` (state assertions only): a fact text that contains any
  of these alternatives does **not** count as a match. Used so a superseding fact that names the
  old value as history ("Sofie moved from Aarhus to Copenhagen", "now called Bobby, formerly
  Bølle") does not trip `facts_absent` / `facts_invalidated` checks on the old value.
- Quote bare `yes`/`no`/`on`/`off` and date-like keywords (`"yes"`, `'2026-11-01'`): YAML 1.1
  parsers turn them into booleans/dates. The validator rejects non-string alternatives.
- Regexes must run in Rust's `regex` crate: no lookaround, no backreferences (validator checks).
- Matchers assume one fact = one claim. Where a system writes compound facts, a mechanical
  failure may be a false alarm; the `assert` text is the rubric to confirm it.

## Assertions on memory state (after a session)

The "memory view" is whatever the system under test exposes as derived facts: today
`current_facts()` (+ closed facts for `scope: ever` / invalidated). If the store becomes an
episode log, the runner must expose the derived fact cache for these checks.

All state assertions also accept `exclude` (see above).

- `facts_present: [{assert, match, max_count?, kind_hint?}]` — at least one **current** fact
  satisfies `match` (one fact must satisfy all groups). `max_count: 1` additionally fails if more
  than one current fact matches (duplicate check). `kind_hint` (user/preference/grace/environment/
  followup) is informational, not asserted.
- `facts_absent: [{assert, match, scope?}]` — no fact matches. `scope: current` (default) looks
  only at current facts; `scope: ever` also at closed/superseded facts (used for garbled input and
  Grace's inventions, which must never have been written at all).
- `facts_invalidated: [{assert, match}]` — at least one closed/superseded fact matches **and** no
  current fact matches.
- `commitments_open: [{assert, match, due?}]` — an open commitment (or, until a commitments table
  exists, a current `followup` fact) matches. `due` (YYYY-MM-DD) is the earliest date it should be
  surfaced; before that date it must not be raised.
- `commitments_done: [{assert, match}]` — a matching commitment exists and is closed (no open/
  current one matches).

`assert` is the natural-language statement; it doubles as the rubric for an optional LLM judge.

## Probes

`probes: [{question, expect?, must_not_contain?, abstain?, proactive?, rubric?}]`

- Asked at the **start of the next session**, at its `start` time, before its turns; for the last
  session at `expect.probe_at`. Each probe is asked independently through the full answer path
  (session block + recall + LLM) and **must not be learned from**: run probes against a copy or in
  a rolled-back transaction, and do not add probe turns to the transcript.
- `expect` — matcher groups (AND of ORs) over Grace's reply.
- `must_not_contain` — list of strings/`re:` regexes; any hit fails.
- `abstain: true` — the reply must contain an abstention marker and nothing from
  `must_not_contain`. Default markers (case-insensitive): `don't know`, `do not know`,
  `not sure`, `haven't told`, `have not told`, `didn't tell`, `never told`, `don't remember`,
  `don't have`, `no record`, `not mentioned`, `haven't mentioned`, `never mentioned`, `no idea`,
  `ved ikke`, `har ikke fortalt`.
- `proactive: true` — `question` is a greeting; checks that Grace raises an open commitment by
  herself. The runner may also assert the commitment is in the session-start context.
- `rubric` — free text for an LLM judge; mechanical checks remain authoritative.

## Isolation

Every scenario starts from an **empty memory** (fresh DB). Names recur across files (Sofie,
Jonas, Kasper) but scenarios are independent and may disagree with each other.

## Clock

- Turn `i`'s user utterance happens at `start + sum(gap_s of turns 0..=i)` (the first turn's
  `gap_s` counts from `start`; default 20 s each). Grace's reply follows 2 s later.
- After the last turn the session **ends** (runner calls whatever end-of-session work exists:
  summaries, reconciliation), then `expect` is checked, then the clock jumps to the next `start`.
- Offsets: `+02:00` until 2026-10-25 03:00, `+01:00` until 2027-03-28 02:00 (validator checks).

## STT confidence

Turns without `stt_garbled` use `stt_logprob: -0.2` unless given. `stt_garbled: true` turns
default to `-0.7` (below the engine's -0.55 drop threshold): a runner that reproduces the engine's
gate may drop them before memory; a runner that tests the memory gate alone passes them through.
Report both if useful. Grace's reply on a garbled turn is what she would have said had the text
reached her.

## Interrupted turns

`interrupted: true` means Grace's reply was cut off by the user; `grace` holds the part that was
spoken. The user's words are complete. Memory must still learn from that user turn (live failure).
