# 0017 — Style rules learned from feedback, and a review queue

- **Status:** Accepted
- **Date:** 2026-10-20

## Context
In live run 4 Mikkel told Grace to give shorter answers and she didn't. The instruction lived
only in the chat history: weak while the session lasted (one user line against a long system
prompt) and gone in the next session. The memory extractor could store "Mikkel prefers short
answers" as a preference fact, but facts are a flat list of what is true about his life, shown
once per session, with no notion of conflict, removal or a cap.

Agreed design for Grace's personality is a hybrid: (a) a persona defined up front by examples
(`grace-chat-bench`, personality branch), (b) runtime style rules learned from "do that / don't
do that", (c) an offline review loop that turns flagged replies into eval scenarios and spec
examples, (d) later a LoRA on curated good replies. This ADR is (b) and (c). Grace must never
rewrite her own core prompt.

## Decision
- **Style rules are their own table** (`style_rules`: text, Mikkel's quoted words, the reply he
  reacted to, source turn, created / invalidated / superseded), not facts. The fact extractor's
  prompt is unchanged: one added sentence telling it to leave talking style to the rules cost
  14 checks on `grace-memory-bench` (220 → 206, mostly unrelated extraction), so a style wish may
  also exist as a preference fact. The rules block is the one that says how to talk.
- **A dedicated classifier** (one JSON call on LLM slot 2 after each exchange, before fact
  learning) sees Grace's previous reply, his words, her answer and the current rules with ids,
  and returns `feedback`, `complaint` and `add` / `update` / `remove` ops. Momentary requests
  ("hold kæft", "sig det igen", "snak lidt langsommere" in one answer) are not rules.
- **Code checks every op** like memory does: the evidence must quote Mikkel (≥ 80 % of words),
  only shown rule ids can be changed, a rule is ≤ 200 characters, at most 10 rules; over the cap
  the oldest retires. A conflicting refinement is an `update` that supersedes the old rule.
  A new or changed rule also needs the classifier's `feedback` flag (without it, it was policing
  Grace on its own), and evidence made only of hush words ("hold lige kæft", "vent") can't make a
  rule. Dropping a shown rule needs only his quoted words.
- **Rendered by code** into a "How Mikkel wants you to talk" block after the memory block,
  oldest first, "the later one wins", each with his words. Read from SQLite every turn, so a
  rule learned on one turn applies from the next and persists across sessions. The block only
  changes when a rule does, so llama-server's cached prefix survives every other turn.
- **Restated on the current user message** for that turn only (`style::with_reminder`, next to
  recall): in the system block alone a "fewer questions" rule held in-session but not in the
  next session (question rate 0.67, same as without rules); with the reminder 0.00.
- **By voice:** she can list the rules (they are in her prompt) and he can drop one ("du må godt
  stille spørgsmål igen" → `remove`). `grace-review rules` / `drop-rule` do the same by hand.
- **Review queue** (`review_queue`, same local DB, gitignored `data/`): every exchange where he
  called a reply bad or a rule changed is stored with the reply, his words and 6 turns of
  context. `grace-review list | show | export | done | dismiss` lets a Claude Code session go
  through it with him and turn entries into `fixtures/chat/scenarios` and examples in
  `config/prompts`. A person decides what becomes a scenario or example; the model never edits
  `system.md`.

## Measurements
`grace-style-bench` (fixtures/style, 7 scenarios, 3 seeds, learning off vs on; docs/MODELS.md):
checks 48/69 → 241/243. Next session after "svar kort": 43 → 21 words; after "stop med at
spørge": question rate 0.67 → 0.00. Feedback turns that changed a rule 18/18; other turns 1/120.
Ten rules at once still give a substantial technical answer (119 words) and pass the
"not overdone" judge 4/4. `grace-memory-bench`: 220 (base, same day) → 225.

## Consequences
- A rule change invalidates the prompt cache once (one slower TTFT); rules change rarely.
- One more background LLM call per exchange (~0.3-1 s on slot 2), never on the voice path.
- A style wish can exist both as a rule and as a preference fact. When he drops a rule by voice
  the fact extractor sees the same words and may close the fact too, but nothing guarantees it. `grace-memory-bench` counts rules as preferences.
- Watch: false rules from ordinary turns (measured: scenario 06), rules too vague to act on, and
  ten rules combining into overdone behaviour (scenario 07). Cap and wording are tuned there.
