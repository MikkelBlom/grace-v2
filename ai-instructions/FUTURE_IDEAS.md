# Future ideas

Parked so they don't derail the current milestone. Promote to a GitHub issue when it's time.

## Speech
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

## LLM
- **[high]** MTP speculative decoding for Gemma 4 (`--model-draft`) in llama-server; measure.
- **[medium]** mistral.rs embedded in-process once its Windows CUDA build is proven.
- **[medium]** DFM-Mimir (Danish 1B) as a rewriter/summariser, not an agent.

## Harness / memory
- **[medium]** Periodic "what's new" scan for better models (v1 "Tech Horizon Scanner"), proposing
  bake-off runs, never auto-switching.
- **[medium]** Save audio per turn next to its trace; one click turns a bad turn into a fixture.
- **[low]** Trace retention/rotation.
