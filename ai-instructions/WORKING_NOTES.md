# Working notes

## Environment / gotchas
- **Bench Python envs** are `uv` venvs with `--system-site-packages`, reusing the global Python
  3.12 CUDA torch (2.11.0+cu128, needed for the 5090 / Blackwell). One venv per conflicting
  candidate: `bench/.venv` (transformers 4.57, sherpa-onnx, supertonic, pyarrow) and
  `bench/.venv-chatterbox` (transformers 5.2, numpy < 2.4). Never `pip install` a package that
  pulls its own torch into these venvs: chatterbox deps pulled a CPU torch 2.14 once and broke CUDA.
  If it happens: `uv pip uninstall --python <venv python> torch torchaudio`.
- **Windows pipes default to cp1252**: Python runners must `reconfigure(encoding="utf-8")`
  (done in `bench/runners/common.py`) or æøå break the JSON protocol.
- **FLEURS ids are per sentence, not per recording**; clip ids add an index.
- **CoRal v3 is gated** (auto-approve): Mikkel must click "request access" on HF.
- **GitHub project board** needs the `project` token scope: `gh auth refresh -s project`
  (interactive, Mikkel runs it).

## Measurements so far (details in docs/MODELS.md)
- Whisper-family STT has a near-fixed cost (encoder runs a padded 30 s window): edda on
  CTranslate2 ≈ 240 ms regardless of clip length. Transducers (ekko) scale with length.
- Chatterbox (Røst 350m) ≈ 2 s per sentence non-streaming; RTF 0.44, so streaming could work.
- Piper ≈ 90 ms per sentence on CPU; quality is the known weak point.

## Open questions
- **Foundations rethink pending Mikkel's go (2026-10-11).** Research in
  `docs/research/2026-10-11-foundations.md` says several post-live-session fixes are symptom
  patches: the "no longer true" prompt list, followup-as-fact-kind, distance dedupe deciding
  merges, the avg_logprob -0.55 cutoff (calibrated on read speech), the hanging-word rule. Proposed
  order: (1) labelled eval sets for STT (own turns) and memory (scripted multi-session), (2) STT
  input fixes (linear AEC output for STT, VAD-trim, fp16 beam 5 temp 0, ekko cross-check),
  (3) memory rebuilt as episode log = truth, facts = derived cache, commitments table, profile
  rendered by code, (4) Smart Turn Danish fine-tune, (5) Røst v3 Chatterbox Danish TTS bake-off.
- **Decided 2026-10-11:** v0.3 paused for v0.2.5 Foundations (ROADMAP). Rule: Mikkel is not
  the test set (CLAUDE.md). Fast fact gate = Gemma logprob classification first, not a new model.
- CoRal v3 access works (2026-10-11). Acoustic replay tier: Grace on one output (Z337), the
  "person" on the other (laptop speakers); only while Mikkel is out of the house, never at night.
- **grace-sim findings (pilot, 40 scenes/set, 2026-10-11):**
  - Echo after warm-up: 0 false triggers (matches live). Cold start (AEC3 unconverged, first
    ~3 s of Grace's first reply): ~12 % of scenes trigger. Scenes now carry a 6 s warm-up.
  - Barge-in while Grace talks: onset median ~1.2 s late, 7/40 not detected at all (talking
    boost + gate). People 1.5-3 m away: 4/40 never trigger VAD.
  - STT (edda int8, prod settings), oracle cuts: clean CoRal 14.3 % WER (matches leaderboard);
    far: raw 43.8 / linear AEC 46.0 / full AEC chain 48.8; double-talk: raw 41.3 / linear 40.3 /
    full 47.4. The full chain (NS + residual suppression) costs 5-7 points. Pilot noise was a
    white-noise fallback (DEMAND not yet downloaded), so absolute far numbers are pessimistic.
  - Sim echo level calibrated to real Z337 recordings in bench-results/aec (mic ≈ loopback −3 dB).
- **Barge-in sweep (grace-sim, 40 double-talk + 40 echo scenes):** the VAD gate's talking boost
  is not the bottleneck (boost 0 → still 0.7 s late, 6/40 missed). VAD on AEC3's linear output
  catches 40/40 but residual echo triggers 104 times per 40 echo scenes. A double-talk detector
  (linear output vs removed echo, ≥ 0 dB while Grace talks) removes cold-start false triggers
  (5 → 0) at the same detection (35/40) but onset stays ~1.5 s. Root cause: a person 0.8-2 m
  away is 5-10 dB quieter at the mic than Grace's echo from speakers 0.3-0.8 m away; AEC3's
  ERLE isn't enough to separate them. Options: lower playback level / speaker placement,
  ducking on any linear-VAD hint, or a neural residual-echo suppressor (trainable; see FUTURE_IDEAS).
- **Live run 3 (2026-10-11, 3D printer running):** speech end → first audio 2-4 s, caused by
  Smart Turn da1 @ 0.5 scoring complete questions 0.06-0.41 (waits), not by models (STT ~370 ms,
  TTFT ~145 ms; Pocket TTS first audio 500-800 ms while other jobs ran). A 300 ms noise during
  Grace's reply cancelled it and the "turn" had no words, so she went silent (now: pause, then
  cancel only on words, else resume). Spelled letters became "O G E E E" and were stored as a
  fact (now: turns with avg_logprob < -0.5 are not learned from; fact 28 invalidated by hand).
- **Memory bench** (`grace-memory-bench`): scenario 01 passes on v0.2 code but the extractor
  invented a "14 December" deadline for the hand-in (matchers can't see compound/invented
  details; rubric/LLM-judge needed for that class).
- System prompt personality was written from live feedback, not v1's `config/personality.json`
  (witty + warm, sharp friend, kind pushback). Merge both.
- AEC option A (Windows comms AEC with render-endpoint reference) may not be exposed by this
  laptop's audio driver; check before building option B.
- E4B (Tau2 42) vs 12B (Tau2 69): published agentic scores favour 12B strongly; decide on our
  own tool traces in v0.3, measuring TTFT too.
- `ekko-v1-tiny` licence is a custom composite (NVIDIA Open Model + CoRal + CC-BY); read before
  depending on it.

- **Live run 4 (2026-10-20, headset, traces/engine-1792475469070.jsonl, 48 answered turns).**
  Speech end → first audio p50 1.38 s, p90 3.66 s, 0/48 under 800 ms (issue #35): turn-end wait
  p50 590 ms (11/48 hit the ~3 s timeout on low Smart Turn scores), Pocket TTS first audio p50
  507 ms (bench: 94 ms), TTFT grows 54 → 355 ms over the session (prefix cache likely invalidated
  each turn). Fixed in the same session: (1) turn audio rings were 30 s, so a 65 s turn lost its
  first 35 s (now 120 s); (2) pausing Grace cut audio in one sample, a click Mikkel heard as
  crackle when he started talking as she did (now a 10 ms fade). Still open: conversation style
  (#36: vague "AI" questions, long replies, doesn't go silent on "hold kæft"), "C#" spoken as
  "køh" (#37). Memory worked: "I have no sister" invalidated the old test fact and added the
  correction. Fact 21 ("semester project deadline 16 November 2026") checked
  against itslearning: 16 Nov ends the *First deployment* phase (12 Oct - 16 Nov); final
  documentation phase is 07-18 Dec (DigitalExam). Not a test fact; the wording is imprecise.
  STT quality in run 4 is not representative: Mikkel walked around, faced away, even closed a
  door, believing the (absent) DJI clip mic was on.
- **Turn length (2026-10-20):** rings stay 30 s between turns, but the conductor pins an open
  turn (`Shared::keep_from`) so it is kept whole (hard cap 30 min). Next limit for long turns is
  STT: every pause re-transcribes the whole turn (~0.2 s per 30 s window), so a 10 min monologue
  would cost seconds per pause. Fix if it bites: transcribe settled chunks once (FUTURE_IDEAS).
- **Docker crashes (Mikkel, 2026-10-20) are not Grace's:** nothing in Grace uses Docker (memory is
  SQLite in-process, data/memory.db). Likely other agents' containers or WSL memory pressure
  (seen before when the Windows Audio service stopped).
- **Bilingual input parked (#34, 2026-10-20):** Mikkel speaks Danish *or* English. edda can't
  detect English (p_da = 1.0 on 60/60) and has 57.7 % WER on English forced to Danish; Whisper
  language ID misroutes conversational Danish (turbo 6/60 whole turns, tiny-small 13-16/60).
  `bench/stt_lang.py`. `stt_lang.py en` crashes natively after ekko loads (not investigated).

- **Conversation style (#36, 2026-10-20):** `grace-chat-bench` + `docs/conversation-style.md`;
  prompt tuned 0.793 → 0.975 (docs/MODELS.md). Run: start llama-server, then
  `cargo run --release --bin grace-chat-bench -- --label <name>` (~7 min for 3 seeds;
  `--no-judge`, `--only 04`, `--system-prompt <file>` for A/B). Findings: (1) asked for an empty
  reply, Gemma loops `<|channel>thought` tokens up to max_tokens (TTS would read them), so the
  prompt says "always at least one word" and quiet = exactly "Okay."; true silence and token
  stripping need engine code (#38, not done: conductor.rs was being changed elsewhere).
  (2) Strong interview rules leak: she started quizzing on "tak" until scoped to "only when he
  asks". (3) The judge is Grace's own model and lenient; it passed leaked tokens as "silent".
  Keep the safety-relevant rules as hard checks. Simulated user is a fictional persona; Grace's
  prompt still mentions Mikkel's real projects (Dokta, Rust), which shows up in transcripts.
  Not verified live yet: the next live run should try "hold kæft", a dictated note, "haha".
- **Persona direction (Mikkel, 2026-10-20):** a Jarvis-like vibe: "a preppy assistant vibe.
  Funny, ironic, and helpful. Talks like a human, not an ai." Chosen by blind preference, not
  described: `bench/style_pick.py` answers 20 short exchanges (`fixtures/chat/pick/`) in each
  candidate style (`fixtures/chat/styles/`: butler, friend, minimal; each swaps only the
  personality paragraph of system.md), scores each on the full bench, and writes
  `bench-results/chat/style-pick/pick.html`. **Waiting on Mikkel's picks**; then fold the
  winning style plus his picked replies (as few-shot examples) into system.md and the spec,
  and re-run the bench. Wit target: about 1 reply in 4; judge has a "trying too hard"
  criterion, and exclamation marks are a hard check.
- **Reply language (2026-10-20):** main's prompt answered casual Danish openers in Danish
  15/15 (scenario 14); a dedicated "Reply language" paragraph fixed it (15/15 English). The
  bench now hard-checks English on every reply. Live runs were English because earlier turns
  were; a cold Danish opener isn't.
- **Memory questions:** Gemma still asks for facts its memory block lists (~1 in 10 interview
  questions), even when told to check first. Likely fix is structural: put the memory block
  before the rules, or render "already known: city, studies" next to the interview rule by code.
- **Hook for learning style rules at runtime** ("don't do that" / "do that"): a separate session
  (`claude/determined-euler-3136d2`) builds it on grace-chat-bench (scenario fields
  `learn_style`, `rules`, `new_session`; `2x-style-*.yaml`). Learned rules should land as a
  rendered block next to the memory block in `system_message`, and be tested with the same hard
  checks so a learned rule can't undo a spec rule (e.g. "be funnier" vs trying-too-hard).

## Decisions made (and why), not yet worth an ADR
- Bench runners are separate processes speaking JSON lines, so any language and any runtime
  can be compared; model load is excluded from per-request latency.
- `grace-debug` polls once a second with byte offsets instead of SSE; simpler, good enough.

## 2026-10-10 (session 2)
- Board: **"Grace v2"** = project #2 (https://github.com/users/MikkelBlom/projects/2), copied from
  v1's board #1 (left untouched). Fields: Status (Todo/In Progress/Testing/Done), Module. Add every
  new issue to it: `gh project item-add 2 --owner MikkelBlom --url <issue url>`.
- CoRal v3 still 403 for HF account `Mikkell` (mikkelblom@live.dk) after Mikkel requested access;
  Pocket TTS (also auto-gated) downloads fine, so CoRal likely needs manual approval.
- `aec_test` plays sound: it requires `--play`. An accidental run played 17 s through the
  speakers once (a `--help` check); never probe it with arbitrary flags.
- sherpa-onnx CUDA wheels need cuBLAS/cuDNN on PATH; prepend `torch/lib` (done in tts.py).
- Devices: speakers = **Logitech Z337**, mic = laptop **Microphone Array (Realtek)**, set in `config/audio.json`. Never use the VF0700 (webcam) or the USB Audio Device. The Realtek "Speakers" are the laptop speakers, not his.
  Headset setup (2026-10-20): output **Hovedtelefoner (JBL Tune 777NC)** (stereo A2DP; the "Headset" endpoints are hands-free, phone quality), mic = **DJI Mic 2** receiver. The DJI receiver showed up only as a USB *drive* ("DJI WIRELESS MIC RX2"), not as an audio input; live run 4 therefore ran on "USB Audio Device" by mistake. config/audio.json now looks for "DJI" / "Wireless Mic" first, so check the startup line for the mic name.
- llama-server pick: `-ngl 99 -c 16384 -np 2 -fa on -ctk q8_0 -ctv q8_0 --reasoning off --swa-full`.
  Without `--swa-full` Gemma 4 pays ~180 ms fixed prefill per request (log warns "cache_reuse is not
  supported by this context"). Don't leave llama-server running after a session: ~10 GB VRAM.
- Two Bluetooth audio devices (Z337 speakers + Buds4 Pro earbuds) fight: with the Buds connected
  and default, opening the Z337 failed with AUDCLNT_E_ENDPOINT_CREATE_FAILED (0x8889000F). Opens
  now retry transient errors and explain them. The engine must also follow device changes
  (IMMNotificationClient) and re-open streams / re-calibrate when the output switches.
- Calibration (2026-10-11, Z337 BT + laptop mic array): playback→mic **110 ms** (105/117/110),
  echo +15.6 dB over room, sweep SNR 52 dB. Independent GCC-PHAT check: 105 ms. The first packet
  of every WASAPI stream is flagged discontinuous; it is not a glitch.
- aec-lab sign bug (fixed): the reference must be shifted so it arrives *before* the echo:
  render[k] = lp[k + offset + lead]. With that, AEC3's estimated delay ≈ lead.
- Pure-Rust aec3 works: synthetic speech echo (50 ms, small IR, noise) → 21–28 dB reduction.
  Sweeps are a poor AEC adaptation signal (one band at a time): use speech runs for AEC tuning.
- VAD acceptance (2026-10-11): our AEC3, adapted + VadGate talking boost → 0.0 s false speech on
  17 s of Grace through the Z337 (raw 15.7 s, Windows AEC 15.2 s). AEC3 needs ~5-10 s of Grace
  speech to converge: the engine must run it continuously and persist nothing yet (re-adapts per
  launch; consider warming it with the calibration/first reply). Double-talk is the weak side:
  near-end speech 6 dB below echo is only partly detected. Next: real double-talk recording
  (Mikkel reads aloud during aec_test playback), AEC3 suppressor tuning.
- AGC2 in the aec3 linear pipeline must stay off for VAD purposes? Mixed: it boosts residual
  (bad for level metrics) but VAD false speech was lower with it (1.7 s vs 3.7 s first pass).
  Decide with real double-talk data.
- **Bluetooth echo delay is per-session and drifts** (bench/echo_delay_track.py): calibration
  110 ms; speech run 113→137 ms within 14 s; YouTube run steady 165 ms. The engine must track
  delay continuously from whatever plays (GCC-PHAT on loopback vs mic, feeding AEC3 a coarse
  alignment and letting its own estimator do the fine part). Calibration = starting point only.
- YouTube + Grace through the Z337: VAD false speech raw 18.8 s, Windows AEC 19.0 s, ours with
  the session's true delay 0.4 s (one blip). TurnMachine's min_barge_in (350 ms) and empty-
  transcript drop would absorb it, but aim for 0.
- Double-talk run 1791682421 (Mikkel reading 5 lines over Grace): VAD on our AEC3 finds his lines
  (~0.5-1 s after each was shown); STT WER raw 18 %, Windows comms 9 %, ours 32-35 %, AEC3 linear
  68 %; Grace's English never appears in the Danish-forced transcript. → ADR 0012 dual path.
  Echo delay this session: ~158 ms (one outlier window at 398 ms during double-talk).
- Smart Turn v3.2: use the **fp32** model (`smart-turn-v3.2-gpu.onnx`, saved as -fp32; runs on CPU
  ~43 ms with 2 threads). The int8 `-cpu` model is numerically unstable near 0.5 (0.26–0.55 on one
  clip across ORT settings). Rust features == HF WhisperFeatureExtractor (1.7e-5); predictions
  identical to Python. Our complete-vs-cut FLEURS test: 70 % at 0.5 (published Danish 92.8 %).
  Tuning idea for TurnMachine: trust a "complete" verdict only after a minimum silence (~300 ms),
  and re-score as silence grows, to cut premature replies (25/60 cut-offs scored "complete").
- bench/.venv: something pulled huggingface_hub 2.2 (breaks transformers 4.57); pinned back to
  0.36. Pin it if installing anything new there.
- Engine architecture as built (2026-10-11): llama-server + speech_server.py hold the models; the
  `grace` engine (0.1 s startup) runs capture ×3 (raw mic, comms mic, loopback), always-on render
  (silence when idle keeps BT awake + loopback continuous), frontend thread (align by QPC clock,
  AEC3, decimate, VAD, gate, GCC delay tracker every 2 s), and the async conductor (TurnMachine,
  Smart Turn + STT on pause, LLM → sentence splitter → TTS → render, barge-in cancel). The
  conversation logic is still inside the engine (conductor); ADR 0002's separate harness process
  is deferred until there is enough harness to warrant it.
- Selftest: `./scripts/grace.ps1 -SelfTest` (silent). Engine-only restart: `cargo run --release -p grace`.
- First live run (2026-10-11, traces/engine-1791683793518.jsonl): speech end → first audio
  0.65–0.75 s (+~0.15 s Bluetooth). Bugs found and fixed: (1) "reply finished" never fired
  because the always-on render keeps the device buffer full → every next utterance counted as
  barge-in; now drained = last sample written + queued buffer time. (2) Phantom turn: a 224 ms
  echo blip passed barge-in (hangover counted as speech) and the comms path turned Grace's
  English echo into Danish words → gate needs 320 ms while Grace talks, and STT uses the AEC path
  whenever the speakers played during the turn (also fixes YouTube leaking into transcripts).
  (3) TTFT rose 60→500 ms from a sliding 20-message window → history grows, pruned in halves at
  5000 tokens. (4) Engine survived Ctrl+C under `cargo run` → launcher runs the exe, q+Enter /
  close / Ctrl+C all stop it, 2 s forced exit, single-instance lock. (5) ort TRACE flood → trace
  layer keeps only grace* targets below WARN.
- Voice: `config/speech.json` tts_voice (default jane); samples in bench-results/voices/tts/listen.html.
  `./scripts/grace.ps1 -Restart` restarts the speech server to apply it.
- Headset run (2026-10-11, traces/engine-1791685410782.jsonl, JBL Tune 777NC): speech end → first
  audio 0.66–0.79 s. Fixes: TurnMachine waits 3 s (not 1.6 s) when Smart Turn scores < 0.1
  ("jeg tænker på at øh" scored 0.005 but was cut off by the fallback); render counts a reply as
  done after its last audible sample (Pocket TTS pads ~0.5 s of silence, so Mikkel's next words
  landed in the padding and showed as "(interrupted)"); launcher port check 200 ms instead of
  Test-NetConnection's ~10 s.
- Memory (v0.2, 2026-10-11): `grace --memory-lab` replays scripted Danish exchanges through the
  real learning path into data/memory-lab.db. With the tightened prompt it stores the exam,
  sister and preference, supersedes the sister's city, invalidates the cancelled exam, and stores
  nothing for small talk. Learning takes 0.5–2 s per exchange in the background.
- bench/.venv-embed has transformers 5.19, sentence-transformers 6.1, torchvision 0.26 (cu128,
  installed --no-deps) and huggingface_hub 2.x; keep it separate from bench/.venv (hub < 1.0).
- **Windows Audio service hangs** (2026-10-11 03:28 and 04:51, same bucket 1925226297557747192,
  AppHangB1 in svchost): both during sessions with a communications-mode stream + echo reference
  on a Bluetooth device. ADR 0013 removed that stream from the engine. If it ever recurs without
  it, look elsewhere (Bluetooth driver, Realtek APO). Recovery: admin `Start-Service Audiosrv`.
- Memory session 2026-10-11 (traces/engine-1791686801317.jsonl): the 9 "to" transcripts all came
  from the comms path; the engine went deaf at 149 s when the audio service hung, and the old
  watchdog (sequential joins) never reported the dead capture thread. Now: self-healing audio
  threads, concurrent watchdog, EndOfTurn so a reply isn't "drained" while TTS is still behind.
  Memory itself worked in that run: sister Aarhus → København corrected and recalled.
- Memory from the live run, cleaned by hand in Mikkel's local data/memory.db (2026-10-11): closed
  fact 7 "Mikkel wants Grace to be quiet" (from a momentary "hold din kæft"), 10 (vague
  inference) and 9 (duplicate of 8). Learning now skips near-duplicates (cosine < 0.08) and the
  prompt says momentary commands are not preferences; memory-lab covers both.
- Live memory runs 2–3 (2026-10-11): the learning step re-stored a closed fact because Grace said
  it (an old session summary still mentioned the exam); a requested reminder was buried in the
  fact list; facts in interrupted turns were never learned; Whisper hallucinated a fluent
  sentence from a breath. Fixes: extraction treats Grace's reply as context only and ignores
  garbled text; new fact kind `followup` shown first ("bring these up"); block lists "no longer
  true" facts (closed in the last 30 days) and labels summaries as background; learning runs
  even when the reply was interrupted; STT transcripts with avg_logprob < -0.55 are dropped
  (FLEURS speech ≥ -0.29, noise ≤ -0.62); turn audio (aec + raw) saved to traces/audio/ for
  diagnosing misheard turns; TurnMachine waits the long timeout when a transcript ends on a
  hanging word (øh, og, men, altså …). Personality guidance moved into config/prompts/system.md.
- Mikkel's local memory cleaned by hand again: closed 14, 16, 17, 22–26; fact 13 → followup.
