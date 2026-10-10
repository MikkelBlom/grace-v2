# 0002 — Split processes by how often they change

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
In v1, every test meant a 2–3 minute relaunch, mostly model loading, so milestones went
untested. Later, Grace must change herself (new tools, tuned prompts) while running. Rust
rebuilds take seconds to tens of seconds, so "recompile everything" can't be the update path.

## Decision
Four layers, each restartable on its own:

| Layer | Holds | Changes | Restart cost |
|---|---|---|---|
| **LLM server** (`llama-server`, child process) | LLM weights + KV cache | Almost never | Model load (seconds) |
| **Engine** (Rust) | Audio I/O, AEC, VAD, turn detection, STT, TTS models | Rarely | Speech-model load |
| **Harness** (Rust) | Turn logic, memory, tool routing, prompts | Often | < 1 s (no models) |
| **Live layer** (files) | System prompt, personality, tool manifests, tools themselves | Constantly | None: watched and reloaded |

Processes talk over local IPC with versioned messages. The harness reconnects to a running
engine and LLM server, so restarting it drops no models and, ideally, no conversation state
(state is persisted in SQLite, [0007](0007-memory-model.md)).

## Consequences
- An IPC protocol to design and version. It also gives us a seam for tests (fake engine).
- Targets: harness restart < 1 s; full cold start dominated by model load, aiming for < 10 s.
- Prompt and tool changes never need a rebuild; that is also the path for self-evolution later.
