# Future ideas

Parked so they don't derail the current milestone. Promote to a GitHub issue when it's time.

## Speech
- **Turn-taking follow-ups (ADR 0018):** Danish articles "en"/"et" as hanging words (a sim
  miss committed on "Danmark er en"); a softer stop when a merge retracts audible speech (fade
  + short "mm"?) if the stop-start feels abrupt live; keep the paused reply's LLM stream alive
  only to the first sentence while reopened; grace-turn-sim scenes with simulated barge-ins
  (person reacting to Grace) and the laptop-speaker echo path.
- **edda LoRA on code-switched Danish** (high): English tech words and spelled letters are
  misheard (24 % WER on the code-switch set) and edda ignores vocabulary prompts. Generate a few
  thousand code-switched sentences (Gemma) × 24 NST voices with tts_da.py + room augmentation from
  grace-sim, LoRA edda, evaluate on codeswitch + CoRal (no regression).
- **Spelling mode**: letters spelled out ("D O C T A") come back as "O G E E E"; needs data too.
- **Neural residual-echo suppressor for barge-in** (high): AEC3 can't separate a person 1-2 m
  away from Grace's louder echo (grace-sim: barge-in ~1.2-1.5 s late, 5-7/40 missed). A small
  model taking (linear AEC output, loopback) → near-end mask, trained on grace-sim scenes
  (CoRal + Grace's voice + room simulation), is a known approach (DTLN-aec class). Measure with
  grace-sim double-talk/echo sets.
- **Cold-start double-talk detector**: `double_talk_db = 0` removes the first-reply false
  triggers in grace-sim at no detection cost; consider enabling once validated live.
- **[high]** Two-pass STT: ekko-v1-tiny partials every ~300 ms for early LLM prefill, edda for
  the final transcript. Needs measurement of how often partial == final.
- **[high]** Stream Chatterbox at token level (emit audio per chunk of speech tokens) or export
  Røst-350m to the Chatterbox-Turbo ONNX graph; non-streaming PyTorch is ~2 s per sentence.
- **[medium]** Supertonic 3 on GPU via onnxruntime-gpu/DirectML; CPU is ~850 ms per sentence.
- **[medium]** Piper for the very first short acknowledgement ("et øjeblik") while a better voice
  synthesises the real answer. Voice mismatch may sound odd: test by ear.
- **[medium]** Gemma 4 E4B/12B native audio input as "audio understanding" (tone, non-speech);
  no published Danish ASR numbers.
- **[low]** Voice verification as an optional layer (sherpa-onnx speaker embeddings).
- **[low]** Port v1's STT correction dictionary + voice-taught corrections.

- Bilingual STT router (#34): language ID with a Danish prior, edda for Danish, turbo for English.
- Incremental STT for long turns: transcribe settled 30 s chunks once, re-transcribe only the tail.
- TTS text normalisation (#37): C#, F#, .NET, URLs, numbers with units.

## LLM
- **[high]** MTP speculative decoding for Gemma 4 (`--model-draft`) in llama-server; measure.
- **[medium]** mistral.rs embedded in-process once its Windows CUDA build is proven.
- **[medium]** DFM-Mimir (Danish 1B) as a rewriter/summariser, not an agent.

## Harness / memory
- **[medium]** Periodic "what's new" scan for better models (v1 "Tech Horizon Scanner"), proposing
  bake-off runs, never auto-switching.
- **[medium]** Save audio per turn next to its trace; one click turns a bad turn into a fixture.
- **[low]** Trace retention/rotation.
