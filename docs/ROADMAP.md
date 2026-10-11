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
- [ ] STT bake-off on CoRal v3 conversation + Mikkel's own recordings (CoRal access pending)
- [ ] Blind listening test of the TTS candidates (Mikkel, ideally others too)

## v0.1 — A live conversation
Talk to Grace in Danish through the speakers and interrupt her.
- [x] Audio I/O in Rust (WASAPI mic + render + loopback, clock-timestamped)
- [x] Echo cancellation bake-off: own AEC3 wins on Bluetooth; dual path ([0012](adr/0012-dual-path-capture.md))
- [x] Silero VAD v6 + Smart Turn v3.2 driving `TurnMachine`
- [x] STT (edda-v0.2 via speech server) with speculative transcription
- [x] `llama-server` with Gemma 4 12B, streaming, prefix caching (`scripts/grace.ps1`)
- [x] Sentence-streamed TTS (Pocket TTS) and always-on playback with ducking and cancel
- [x] Models live in separate servers; engine restarts in ~0.1 s (harness split comes with v0.2+)
- [ ] Media-playback false-trigger fixture (must be 0) and latency budget check
- [ ] Live test: speech end → first audio measured on real turns
- [ ] Continuous delay tracking verified live (tracker built, issue)
- [ ] Speaker calibration on request (#32)

## v0.2 — Memory that persists
- [x] SQLite schema: facts (bitemporal, provenance, confidence), sessions + turns (`grace-memory`)
- [x] Embedder bake-off: harrier-oss-v1-270m wins (Danish question → English fact, R@1 0.81)
- [x] FTS5 + sqlite-vec hybrid retrieval with RRF; recall attached to the user's turn
- [x] Session summaries (written at the next start for unsummarised sessions), loaded into the prompt
- [x] Background learning (add/update/invalidate, JSON-schema constrained) on LLM slot 2
- [ ] Live test: Grace remembers across two sessions

## v0.2.5 — Foundations (v0.3 paused until this is done)
Research: [docs/research/2026-10-11-foundations.md](research/2026-10-11-foundations.md).
Rule: no tuning or testing on Mikkel; public corpora, synthetic data and automated harnesses.
- [ ] `grace-sim` injection tier: WAVs + loopback reference fed into the frontend faster than
      real time; RIR/noise/echo mixing; WER, end-of-turn F1, false triggers, latency
- [ ] STT eval sets: CoRal v3 conversation test, Common Voice da, FLEURS da, non-speech (MUSAN)
- [ ] STT input fixes measured on those sets (linear AEC output, VAD trim, decoding settings,
      ekko cross-check)
- [ ] Memory benchmark: synthetic Danish multi-session scripts (persona + dated timeline with
      updates and commitments), deterministic DB assertions + probe questions
- [ ] Memory redesign: episode log = truth, facts = derived cache, commitments table, per-turn gate
- [ ] Smart Turn Danish fine-tune (v3.2 data + Danish synthetic + CoRal cuts)
- [ ] Danish TTS bake-off: Røst v3 Chatterbox 350M/500M
- [ ] Acoustic replay tier (second speaker in the room) and simulated-user tier (nightly)

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
