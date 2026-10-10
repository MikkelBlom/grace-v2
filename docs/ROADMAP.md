# Roadmap

Each milestone ends with a **live test with the speakers on** and a trace that looks right
([0004](adr/0004-testing-strategy.md)). Day-to-day status lives on the GitHub board; this file
holds the shape.

## v0.0 — Foundation
Tooling that every later step leans on.
- [x] Cargo workspace, CI (fmt, clippy, tests on Windows)
- [x] `grace-trace`: non-blocking JSONL tracing with per-turn tagging
- [x] `grace-debug`: live per-turn latency waterfall
- [x] `grace-core::turn`: turn-taking state machine with fake-clock tests
- [x] `grace-bench`: language-agnostic bake-off harness (STT, TTS, blind listening page)
- [x] First STT bake-off (FLEURS-da): edda-v0.2, Røst-v3, ekko-v1-tiny
- [x] First Danish TTS bake-off: Piper, Supertonic 3, Røst-v3 Chatterbox 350m/500m
- [ ] STT bake-off on CoRal v3 conversation + Mikkel's own recordings
- [ ] Blind listening test of the TTS candidates (Mikkel, ideally others too)

## v0.1 — A live conversation
Talk to Grace in Danish through the speakers and interrupt her.
- [ ] Audio I/O in Rust (WASAPI mic + render + loopback)
- [ ] Echo cancellation bake-off A/B/C and integration ([0006](adr/0006-echo-cancellation.md))
- [ ] Silero VAD v6 + Smart Turn v3.2 driving `TurnMachine`
- [ ] STT adapter (pick from bake-off) with speculative transcription
- [ ] `llama-server` child process, Gemma 4 E4B, streaming, prefix caching
- [ ] Sentence-streamed TTS adapter and playback with ducking and cancel
- [ ] IPC between engine and harness; harness restart without losing models
- [ ] Media-playback false-trigger fixture (must be 0) and latency budget check
- [ ] Live test: speech end → first audio measured on real turns

## v0.2 — Memory that persists
- [ ] SQLite schema: facts (bitemporal, provenance, confidence), episodes
- [ ] Embedder bake-off (harrier-oss-v1-270m vs EmbeddingGemma 2) on Danish recall
- [ ] FTS5 + sqlite-vec hybrid retrieval with RRF
- [ ] Session summary written at end of conversation, recalled at the next start
- [ ] Background fact extraction (ADD/UPDATE/INVALIDATE/NOOP), off the voice path

## v0.3 — First tools
- [ ] Tool manifest + JSON-RPC boundary with hot reload ([0010](adr/0010-tool-boundary.md))
- [ ] Web search (local SearXNG first)
- [ ] Open program
- [ ] File search (live index of chosen roots)
- [ ] Task actor with spoken progress, results delivered at a pause ([0008](adr/0008-conversation-and-tasks.md))
- [ ] LLM bake-off on our own tool traces: Gemma 4 E4B vs 12B vs Qwen3.5-9B

## v0.4+ — The harness
Make it get better over time: prompt and personality as live files, tool routing, eval suite
from recorded turns, observed-world indexer (folders, programs, dev servers), consolidation,
self-written tools in a sandbox, periodic "what's new" model scan.

## Later
Overlay/HUD, voice verification as an option, diarization, integrations, everything else from
v1's idea list, one at a time.
