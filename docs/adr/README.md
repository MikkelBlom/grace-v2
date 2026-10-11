# Architecture Decision Records

One short file per decision: what we decided, why, and what it costs. When a decision changes,
write a new ADR that supersedes the old one and mark the old one `Superseded by NNNN`. Don't
edit history.

| # | Decision | Status |
|---|---|---|
| [0001](0001-rewrite-in-rust.md) | Rebuild as Grace v2 with a Rust core | Accepted |
| [0002](0002-process-split-and-hot-reload.md) | Split processes by how often they change | Accepted |
| [0003](0003-observability-first.md) | Trace every stage of every turn from day one | Accepted |
| [0004](0004-testing-strategy.md) | Pure core, recorded fixtures, measured budgets | Accepted |
| [0005](0005-turn-taking.md) | VAD + turn-end model + speculative STT | Accepted |
| [0006](0006-echo-cancellation.md) | Cancel all PC audio via loopback reference | Accepted |
| [0007](0007-memory-model.md) | Remembered vs observed memory, SQLite as truth | Accepted |
| [0008](0008-conversation-and-tasks.md) | Conversation and tasks are separate actors | Accepted |
| [0009](0009-models-by-bake-off.md) | Choose models by bake-off on this machine | Accepted |
| [0010](0010-tool-boundary.md) | Tools live behind a process boundary | Accepted |
| [0011](0011-reply-language.md) | Danish in, English out is acceptable | Accepted |
| [0012](0012-dual-path-capture.md) | Two capture paths: AEC3 for *when*, Windows processing for *what* | Accepted |

Template: [0000-template.md](0000-template.md).
