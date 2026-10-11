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
- AEC option A (Windows comms AEC with render-endpoint reference) may not be exposed by this
  laptop's audio driver; check before building option B.
- E4B (Tau2 42) vs 12B (Tau2 69): published agentic scores favour 12B strongly; decide on our
  own tool traces in v0.3, measuring TTFT too.
- `ekko-v1-tiny` licence is a custom composite (NVIDIA Open Model + CoRal + CC-BY); read before
  depending on it.

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
