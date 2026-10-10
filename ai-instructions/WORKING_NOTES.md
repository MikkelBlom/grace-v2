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
- Default render device was Mikkel's Buds4 Pro earbuds at the time; speakers are
  "Speakers (Realtek(R) Audio)". Default comms mic is "USB Audio Device".
- llama-server pick: `-ngl 99 -c 16384 -np 2 -fa on -ctk q8_0 -ctv q8_0 --reasoning off --swa-full`.
  Without `--swa-full` Gemma 4 pays ~180 ms fixed prefill per request (log warns "cache_reuse is not
  supported by this context"). Don't leave llama-server running after a session: ~10 GB VRAM.
