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

## 2026-10-11 — Claude Code (Opus 5.5): v0.2.5 foundations, first pass
**Summary:** Built `grace-sim` (real frontend on simulated rooms from CoRal + Grace's voice,
echo calibrated to real Z337 recordings, AEC warm-up) and `grace-memory-bench` (40 Danish
multi-session scenarios, fake clock, real answer path). STT now reads AEC3's linear output
(ADR 0014). Memory redesign (ADR 0015): 84 % → 88 %. Merged v1's personality spec into the
prompt. Parallel agents: Smart Turn Danish fine-tune and Danish TTS bake-off (running).
**Issues encountered:** network saturated by the 41 GB Smart Turn dataset (MUSAN mirror ~0.3
MB/s; ESC-50 moved to an HF mirror); embedding-distance dedupe merged "15 October" with
"16 October" and lost a correction (fixed with specifics check); prompt edits mid-run
contaminate a memory bench run (it reads the prompt live).
**Next steps:** full STT scene sets once noise corpora land; barge-in; reconciliation pass.

## 2026-10-11 — Claude Code (Opus 5.5): v0.2.5 foundations, second pass
**Summary:** Full grace-sim STT sets (150/set, DEMAND noise) confirmed linear AEC for STT.
Found edda hallucinates on 100 % of non-speech; replaced the logprob filter with an ekko
cross-check (ADR 0016). Memory: number words, unknown-id updates, commitments+facts (216/246);
reconciliation pass measured and left off. Smart Turn da1 live (60 → 75 % on CoRal). Danish TTS
Røst 500m with CoRal Mic/Nic: ~150 ms on idle GPU. Replay tier built. Followup facts migrated
into commitments in Mikkel's real DB.
**Issues encountered:** GPU/RAM saturation from parallel tracks (latencies invalid while
training); MUSAN mirror failed at 0.96/11 GB (dropped); DEMAND env names lost in the HF mirror;
DEMAND licence may be CC BY-SA upstream (unverified).
**Next steps:** live session with the new stack; replay tier when Mikkel is away; edda latency
on idle GPU; Danish voice decision; barge-in.

## 2026-10-11 (late) — Claude Code (Opus 5.5): live run 3 fixes
**Summary:** Commit d55f379 (logged after the fact): talking over Grace pauses her, and she is
cancelled only if the new turn has words; Smart Turn da1 thresholds 0.5 → 0.3 (long wait 0.05);
memory skips turns with avg_logprob < -0.5; code-switch STT set.

## 2026-10-20 — Claude Code (Opus 5.5): headset live run 4, GitHub cleanup
**Summary:** Mikkel: Grace always answers in English; he speaks Danish or English, but bilingual
input is parked (#34, measured: edda can't detect English). Closed 15 finished issues (#4-#9,
#11-#16, #31, #33), commented on #28 and #32, opened #34-#37. Headset config (JBL stereo out).
Live run 4 with Mikkel: p50 1.38 s speech end → first audio (#35), conversation-style problems
(#36), C# pronunciation (#37). Fixed: 30 s turn-audio ring that cut long turns (now 120 s), and
a click when Grace pauses (10 ms fade, unit-tested).
**Issues encountered:** the run used "USB Audio Device", not the DJI mic: the DJI receiver only
enumerates as a USB drive. Mikkel's spoken note to "the other AI" was cut by the 30 s ring;
recovered from the trace's partial transcripts. `stt_lang.py en` crashes natively (skipped).
**Next steps:** get the DJI to show up as an audio input; short live check of the click fix and
long turns; then #35 latency (Smart Turn waits, Pocket TTS first audio, TTFT growth).
