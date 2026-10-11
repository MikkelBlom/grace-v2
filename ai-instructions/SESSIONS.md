# Sessions

## 2026-10-10 — Claude Code (Opus 5.5): status of v1, v2 foundation (v0.0)
**Summary:** Audited Grace v1 (Documents/Claude/Projects/AI automation/grace): turns 17–40 s,
non-streaming JSON tool loop, ~100 tools, untested features; lessons in docs/LESSONS-FROM-V1.md.
Agreed with Mikkel on a ground-up v2 in Rust in this folder (old spec HTML + empty .slnx moved to
the Recycle Bin). Researched the Oct 2026 model landscape (STT, TTS, LLM, embeddings, AEC,
turn detection). Built the v0.0 foundation: workspace + CI, grace-trace, grace-debug,
grace-core turn machine, grace-bench + Python runners, FLEURS fixtures. Ran the first STT and
Danish TTS bake-offs (docs/MODELS.md). Wrote ADRs 0001–0010, roadmap, README, created the
public repo MikkelBlom/grace-v2 with issues/milestones.

**Issues encountered:** CoRal v3 is gated (needs Mikkel's HF access request). Chatterbox deps
pulled a CPU torch into its venv (fixed by uninstalling it). Windows cp1252 pipes broke æøå in the
runner protocol (fixed). The `gh` token lacks the `project` scope, so the board needs
`gh auth refresh -s project`.

**Next steps:** CoRal + own-voice STT fixtures; blind TTS listening test; then v0.1 audio engine
starting with the AEC bake-off.

## 2026-10-10 (later) — Claude Code (Opus 5.5): reply language, voices, AEC probe, LLM latency
**Summary:** Mikkel chose Gemma 4 12B and accepted English replies to Danish speech (ADR 0011).
Built the "Grace v2" project board (#2, copied from v1's #1, which stays untouched) with all
issues. Fixed the listening page export (results rendered off-screen; ratings now persist).
English TTS bake-off: Pocket TTS ~94 ms first audio on CPU, Kokoro ~250 ms on CUDA. grace-audio:
WASAPI capture/render/loopback wrappers, AEC probe (all 3 mics support Windows AEC with a chosen
reference), echo-reduction test `aec_test --play`. grace-llm: llama-server streaming client and
sentence splitter. grace-bench `llm`: Gemma 4 12B QAT on llama.cpp b11539 with `--swa-full`
reaches TTFT 53 ms, first speakable piece 239 ms, 20/20 tool choice, ~9.9 GB VRAM.

**Issues encountered:** CoRal v3 still 403 for Mikkel's HF account. A `--help` probe of
aec_test played 17 s of audio through the speakers (tool now needs `--play`). Gemma's SWA cache
disabled prefix reuse until `--swa-full`. sherpa-onnx CUDA needs torch/lib on PATH.

**Next steps:** Mikkel runs `aec_test --play` and the English listening test. Then VAD v6 +
Smart Turn adapters, engine process with the turn machine, first live loop.

## 2026-10-11 — Claude Code (Opus 5.5): echo cancellation, VAD, Smart Turn, the engine
**Summary:** Measured that Windows AEC fails on the Bluetooth Z337 (~4 dB); built speaker
calibration (110 ms delay), proved our pure-Rust AEC3 (21–28 dB synthetic), found BT delay
varies per session and drifts (110/113→137/165 ms). Silero VAD v6 + gate: 0.0 s false speech on
Grace's echo after adapted AEC3 + talking boost. Double-talk with Mikkel's real voice led to
ADR 0012 (AEC3 path for *when*, Windows-processed path for *what*; Danish STT ignores English
echo). Smart Turn v3.2 fp32 (int8 unstable). Built services/speech_server.py and the `grace`
engine; silent self-test passes, startup 0.1 s.
**Issues encountered:** a sign error in aec-lab's reference shift; WASAPI loopback silence gaps;
Windows Audio service stopped once (system issue, WSL memory pressure); huggingface_hub upgrade
broke transformers in bench venv (pinned).
**Next steps:** first live conversation (Mikkel), read its traces, tune turn-taking and latency.

## 2026-10-11 — Claude Code (Opus 5.5): foundations research
**Summary:** Mikkel asked whether the memory/listening fixes treat causes or symptoms. Three
literature reviews (memory systems, Danish far-field STT, local fine-tuning) recorded in
`docs/research/2026-10-11-foundations.md`. Found v1's personality spec in `config/personality.json`.
**Issues encountered:** none; no code changed.
**Next steps:** Mikkel decides on the proposed order (eval sets first). See WORKING_NOTES.

## 2026-10-11 — Claude Code (Opus 5.5): fact gates and test data
**Summary:** Researched "Jev"-style fast fact gates and data sources that don't need Mikkel.
Added v0.2.5 Foundations to the roadmap (v0.3 paused) and the "not the test set" rule to CLAUDE.md.
**Next steps:** build `grace-sim` injection tier and the STT eval sets first.
