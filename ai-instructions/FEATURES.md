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
| CoRal v3 fixtures | Implemented | Conversation test via `bench/fetch_corpora.py coral`; train splits for Smart Turn | 2026-10-11 | HF access | — |
| Own-voice fixtures | Planned | Mikkel's recordings through speakers + mic | — | recorder script | — |
| `grace-audio` WASAPI | Implemented | Capture (raw / communications+AEC reference / loopback), render, resample | 2026-10-10 | — | — |
| AEC probe + test | Implemented | `aec_probe` (capabilities), `aec_test --play` (echo reduction) | 2026-10-10 | grace-audio | Mikkel runs --play |
| `grace-llm` | Implemented | llama-server SSE client (content/reasoning/tool deltas), sentence splitter | 2026-10-10 | — | — |
| `grace-bench llm` | Implemented | TTFT, first speakable piece, decode speed, tool-choice accuracy | 2026-10-10 | grace-llm | — |
| Runtime fetch | Implemented | `scripts/fetch_runtime.py`: pinned llama.cpp b11539 + Gemma 4 12B QAT | 2026-10-10 | — | — |
| Speech server | Implemented | `services/speech_server.py`: edda STT + Pocket TTS over local TCP, cancellable streaming | 2026-10-11 | bench venv | — |
| Engine `grace` | Implemented, live-tested (runs 1-4) | Dual-path capture, AEC3 + delay tracking, VAD gate, Smart Turn, conductor, sentence-streamed replies, barge-in, traces | 2026-10-11 | all crates | — |
| Launcher | Implemented | `scripts/grace.ps1` (+ `-SelfTest`) | 2026-10-11 | — | — |
| Harness split | Planned | Conversation logic in its own process (now inside the engine's conductor) | — | engine | — |
| Memory | Implemented (ADR 0015), not yet re-tested live | Code-rendered block, commitments table, quoted evidence, scope, calendar; 216-217/246 on the bench | 2026-10-11 | embed server | — |
| Embed server | Implemented | `services/embed_server.py` (harrier-270m, CPU, :7812) | 2026-10-11 | bench/.venv-embed | — |
| Tools | Planned (v0.3) | ADR 0010: web search, open program, file search | — | harness | — |
| `grace-sim` | Implemented | Real frontend on simulated rooms (CoRal + Grace's voice + DEMAND/ESC-50), ~35× real time; VAD/gate/DTD switches | 2026-10-11 | corpora | — |
| `grace-memory-bench` | Implemented | 40 Danish multi-session scenarios, fake clock, real answer path; `--reconcile` experimental | 2026-10-11 | llama + embed | — |
| Style rules (ADR 0017) | Implemented, not yet used live | Feedback on how Grace talks becomes quoted, capped (10) rules in their own block; listed/removed by voice; taken out of fact extraction | 2026-10-20 | llama slot 2 | — |
| Review queue + `grace-review` | Implemented | Complaints and rule changes logged with context in data/memory.db; list/show/export/done/dismiss, rules/drop-rule | 2026-10-20 | — | — |
| `grace-style-bench` | Implemented | fixtures/style: learning off vs on, before/after/next-session length and questions, false-rule rate, pile-up judge | 2026-10-20 | llama | — |
| STT on linear AEC | Implemented (ADR 0014) | STT reads AEC3 linear output; turn audio saves lin/raw/aec | 2026-10-11 | — | — |
| Hallucination check | Implemented (ADR 0016) | ekko-v1-tiny on CPU alongside edda; drop when ekko hears nothing | 2026-10-11 | speech server | — |
| Smart Turn da1 | Implemented | Danish fine-tune, 60 → 75 % on CoRal conversation | 2026-10-11 | — | live check |
| Danish TTS (Røst 500m) | Measured, not wired | ~150 ms first audio on idle GPU with CoRal Mic/Nic voices | 2026-10-11 | — | Mikkel's listening test, live GPU-contention test |
| Synthetic Danish speech | Implemented | `bench/synth/tts_da.py`: 24 NST (CC0) voices via MIT Chatterbox | 2026-10-11 | .venv-roest | — |
| Acoustic replay tier | Built, not run | `bench/sim/replay.py`: second speaker talks to live Grace | 2026-10-11 | — | Mikkel away from home |
| English-input bench | Implemented | `bench/stt_lang.py` (language ID + English WER), FLEURS en fixtures | 2026-10-20 | — | — |
| Bilingual input (da or en) | Planned (#34) | Route English speech to an English-capable STT | — | language ID | parked by Mikkel |
| Headset config | Implemented | JBL stereo out + DJI mic in, fallback Z337 + laptop mic | 2026-10-20 | — | DJI must enumerate as audio |
