# Features

| Feature | Status | Description | Date | Depends on | Blocked by |
|---|---|---|---|---|---|
| Workspace + CI | Implemented | Cargo workspace, fmt/clippy/test on windows-latest | 2026-10-10 | — | — |
| `grace-trace` | Implemented | Non-blocking JSONL tracing, per-turn tagging, explicit flush | 2026-10-10 | — | — |
| `grace-debug` | Implemented | Live per-turn waterfall, "speech end → first audio" KPI, process log | 2026-10-10 | grace-trace | — |
| Turn machine | Implemented | `grace_core::turn::TurnMachine`: speculative STT, duck-before-cancel barge-in | 2026-10-10 | — | — |
| `grace-bench` STT/TTS | Implemented | JSON-lines runner protocol, WER/CER, latency percentiles, blind listening page | 2026-10-10 | — | — |
| Bench runners | Implemented | HF Whisper, CTranslate2 Whisper, sherpa transducer, Piper, Supertonic, Chatterbox | 2026-10-10 | bench venvs | — |
| FLEURS-da fixtures | Implemented | 60 clips via `bench/fetch_fixtures.py` | 2026-10-10 | — | — |
| CoRal v3 fixtures | Planned | Conversation + read-aloud clips | — | — | Mikkel must request HF access |
| Own-voice fixtures | Planned | Mikkel's recordings through speakers + mic | — | recorder script | — |
| `grace-audio` WASAPI | Implemented | Capture (raw / communications+AEC reference / loopback), render, resample | 2026-10-10 | — | — |
| AEC probe + test | Implemented | `aec_probe` (capabilities), `aec_test --play` (echo reduction) | 2026-10-10 | grace-audio | Mikkel runs --play |
| `grace-llm` | Implemented | llama-server SSE client (content/reasoning/tool deltas), sentence splitter | 2026-10-10 | — | — |
| `grace-bench llm` | Implemented | TTFT, first speakable piece, decode speed, tool-choice accuracy | 2026-10-10 | grace-llm | — |
| Runtime fetch | Implemented | `scripts/fetch_runtime.py`: pinned llama.cpp b11539 + Gemma 4 12B QAT | 2026-10-10 | — | — |
| Audio engine | Planned (v0.1) | VAD, Smart Turn, STT, TTS adapters wired to TurnMachine | — | grace-audio | — |
| Harness | Planned (v0.1) | Conversation actor, llama-server client, IPC | — | engine | — |
| Memory | Planned (v0.2) | ADR 0007 | — | harness | — |
| Tools | Planned (v0.3) | ADR 0010: web search, open program, file search | — | harness | — |
