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

## 2026-10-20 — Claude Code (Opus 5.5): recoverable turn ends (branch claude/turn-taking-b1ffdc)
**Summary:** Jumping in early is now cheap: speech right after a commit reopens the same turn
(re-transcribed whole, half-answer retracted), replies are prepared at uncommitted pauses with
audio held, history/memory hold only what played (render counts samples per reply). New
grace-turn-sim (real VAD/Smart Turn/STT on 250 CoRal turns, virtual time) reproduces live run 4
and measured before/after; thresholds retuned (fallback 1.0 s, long wait 2.0 s, VAD hangover
128 ms). ADR 0018, MODELS.md, live-check list in docs/research/2026-10-20-turn-taking-live-check.md.
**Issues encountered:** ADR 0017 is taken by the style-learning branch; conductor.rs overlaps
with it (one line in speak_reply). min_commit_silence measured no effect (kept at 0).
**Next steps:** Mikkel's live check (DJI + JBL); then the pipeline latency (Pocket TTS first
audio 500 ms live vs 94 bench, STT ~300 ms) since the policy is no longer the bottleneck.


## 2026-10-20 — Claude Code (Opus 5.5): style rules + review queue (ADR 0017)
**Summary:** Side session of the personality work, branch claude/determined-euler-3136d2.
Mikkel's feedback on how Grace talks becomes style rules: a classifier after each exchange
(background), code guards (quoted evidence, feedback flag, hush words, ≤ 200 chars, cap 10),
rendered by code into the system message and restated on the current user message. Voice can list
and drop rules. Complaints and rule changes go to a local review queue; `grace-review` lists,
exports and closes entries. New `grace-style-bench` (fixtures/style, 7 scenarios): checks 48/69
off → 241/243 on; next-session words after "svar kort" 43 → 21; question rate after "stop med at
spørge" 0.67 → 0.00; false rules 1/120. Memory bench 220 (base, same day) → 225.
**Issues encountered:** first version changed the fact extractor's prompt and lost 14 memory
checks (reverted). Built a separate bench rather than extending grace-chat-bench, which was
uncommitted and moving in the personality branch. llama-server stopped mid-run (personality
session's); restarted one, then stopped it. An over-broad `taskkill` on python.exe also killed
four python processes that were not mine (PIDs 37248, 56252, 29120, 70036); siblings told.
**Next steps:** merge order (turn-taking, personality, this) is Mikkel's call; then a live run
where he gives style feedback and checks it next session; first `grace-review export` review
with him.

## 2026-10-20 — Claude Code (Opus 5.5): conversation-style bench (#36)
**Summary:** Style spec `docs/conversation-style.md` (do/don't pairs from live run 4).
`grace-chat-bench` (grace-engine bin): 13 scenarios in `fixtures/chat/scenarios/`, simulated
Danish user (fictional persona), hard checks + Gemma judge, transcripts in
`bench-results/chat/<label>/`. `memory::session_block` made public so the bench renders memory
exactly like the engine. Prompt tuned in 4 rounds: score 0.793 → 0.975 (repeat 0.980); quiet
3/21 → 21/21. Issue #38 for engine-side silence and channel-token stripping.
**Issues encountered:** Gemma emits `<|channel>thought` loops when asked for an empty reply;
over-strong interview rules made Grace quiz on "tak" (fixed by scoping).
**Next steps:** live check of quiet/dictation/fillers/interview; #38 in conductor.rs once the
turn-taking work lands; add scenarios from each new live run's failures.

## 2026-10-20 (later) — Claude Code (Opus 5.5): persona candidates and blind pick
**Summary:** Persona direction from Mikkel ("preppy assistant vibe, funny, ironic, helpful,
talks like a human"). Spec rewritten around example exchanges. Three candidate styles
(butler, friend, minimal) + current as control, scored on the bench and answered on 20 short
pick exchanges; blind pick page `bench-results/chat/style-pick/pick.html`. Bench: "trying too
hard" judge criterion, exclamation-mark check, multi-scenario YAML files, transcripts.jsonl.
**Issues encountered:** llama-server was shared with another session (theirs, left running);
the judge's old rubric penalises "sir".
**Next steps:** Mikkel picks; fold winning style and his picked replies into system.md as
few-shot examples; re-run the bench; fix the memory-question and "That sounds like" regressions;
"What is your name?" when she knows he's Mikkel (interview order should say full name).
**Later the same day:** reply-language scenario 14 + English hard check (main's prompt: 0/15
English on cold Danish openers; fixed: 15/15). Interview rule reworded (no example questions
that memory usually answers, "full name, you know he's Mikkel").

## 2026-10-20 (afternoon) — Claude Code (Opus 5.5): Mikkel's replies as ground truth
**Summary:** Round-1 style pick rejected (generated candidates "VERY shit made"). His own
replies became the few-shot block in system.md and the examples in docs/conversation-style.md;
judge re-anchored on them. Prompt fixes: reminders are possible via memory, greetings get a
greeting back, no "That sounds like" openers. Round-2 pick page: 16 hand-written exchanges,
two hand-written candidates + Grace's reply, blind, with a box for his own reply
(bench-results/chat/style-pick-2/pick.html). Bench 0.898 under the new rubric.
**Issues encountered:** two Claude crashes mid-run; the first bench after the crash used a
stale binary (clippy doesn't rebuild), re-run. The bench loaded the candidates YAML as a
scenario (moved to fixtures/chat/pick-handwritten.yaml).
**Next steps:** Mikkel does round 2; fold his picks and written replies in; small talk still
interviews him; known-fact questions; live check.
**Round 2 folded in (same day):** 16 more of his picks/notes as examples in system.md and the
spec; scenario 15 (memory-aware opinions, from his "I hate python" note). Bench 0.909.
**Principles instead of pairs (same day):** prompt's 24 example pairs → "Your manner" rules + 7
illustrations. New held-out set (22 lines) + `bench/example_reuse.py` + `bench/reaction_openers.py`.
Held-out 0.965 → 0.995, reuse 9 → 3 of 66; full bench 0.909 → 0.929.


## 2026-10-20 (evening) — Claude Code (Opus 5.5): merged the three side branches into main
**Summary:** On Mikkel's go, merged turn-taking (claude/turn-taking-b1ffdc), style rules
(claude/determined-euler-3136d2) and personality (claude/personality-eval-304546) into main, in
that order. conductor.rs auto-merged (prompt_block + rule reminder sit on the turn-taking
rewrite; `memory.learn` runs at reply end and includes `learn_style`). Conflicts were only in
notes, MODELS.md and the ADR index (kept both sides). fmt, clippy, workspace tests and the
release build pass.
**Issues encountered:** none in code. The style bench was measured on main's old system.md and
the chat bench without learned rules; the combination is not benched yet.
**Next steps:** live session with DJI + JBL (docs/research/2026-10-20-turn-taking-live-check.md
plus style feedback, then check it holds the next session); `rules:` field in grace-chat-bench.

## 2026-10-21 — Claude Code (Opus 5.5): live run 5 on merged main
**Summary:** Live test on main after merging turn-taking, style rules and personality (DJI Mic 2
-> JBL Tune 777NC). Latency p50 1.28 s / p90 2.04 s (run 4: 1.38 / 3.66); merging continuations
works (23 reopens, half-answers retracted). Findings in WORKING_NOTES "Live run 5".
**Issues encountered:** a >30 s reopened turn lost its start (ring unpinned on Respond); resumed
stories restart the cut sentence; style learning stored 5 poor rules + junk facts and a
"continue the story" commitment in the real DB; invented movie facts.
**Next steps:** Mikkel decides which fixes to do and whether to drop the learned rules/facts.
**Fixes the same day (2026-10-21):** long reopened turns keep their start (ring pinned until
the reply finishes); memory cleaned (5 rules, facts 50-56, commitment 2); style classifier must
name a rule's category and code keeps only manner (style bench 08: 17/27 -> 27/27, full: 18/18
learned, 0/144 false); honesty rule for unknown films/trivia (scenario 16) and no question
tails or compliance announcements (chat bench 0.942 -> 0.956). #39 filed (exact-word resume),
#36 closed. Echo was untested (headset).

## 2026-10-21 — Claude Code (Opus 5.5): tools discussion + JevK5 bake-off
**Summary:** Tools direction agreed (proactive assistant; free search via SearXNG + fallbacks;
looping lookups; a Jev-style checker). Bake-off of JevK5 4B vs 9B with Gemma as baseline on 178
hand-written decisions (`bench/decide_bench.py`, `fixtures/decide/scenarios.yaml`): 4B 147/154,
Gemma 146, 9B 145; 4B 3.5 GB, 9B 6.2 GB; ~130-145 ms per decision idle, ~210-260 ms while Gemma
generates; a busy checker slows Gemma 50 -> 33 tok/s. Errors of Gemma and the 4B overlap on 1.
**Next steps:** Mikkel decides whether JevK5 4B becomes the checker; v0.3 tool work (#17, #18).
