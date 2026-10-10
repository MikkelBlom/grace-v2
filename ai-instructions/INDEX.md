# AI notes index

Working notes for any AI agent on Grace v2. Polished design lives in `docs/` (ADRs, roadmap,
model register); these files are the scratchpad around it. Keep them short.

| File | What | Update when |
|---|---|---|
| `../CLAUDE.md` | Project identity and hard rules | Fundamentals change (rarely) |
| `GLOBAL-CLAUDE.md` | Copy of Mikkel's global AI guide | His global guide changes |
| `FEATURES.md` | What exists vs planned, by crate | A feature lands or changes status |
| `WORKING_NOTES.md` | Open concerns, decisions in flight, gotchas | Continuously; prune when resolved |
| `FUTURE_IDEAS.md` | Ideas parked so they don't derail current work | Whenever one comes up |
| `SESSIONS.md` | One entry per work session | End of every session |

Where things are tracked:
- **Decisions** → `docs/adr/`.
- **Model picks and numbers** → `docs/MODELS.md` (raw results: `bench-results/summary.jsonl`, local only).
- **Milestones and tasks** → GitHub issues/milestones/board on `MikkelBlom/grace-v2`;
  `docs/ROADMAP.md` holds the shape.
