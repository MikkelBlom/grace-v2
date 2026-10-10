# Models: current picks and bake-off results

Living register for [0009](adr/0009-models-by-bake-off.md). Numbers are from `grace-bench` on
the RTX 5090 Laptop (24 GB) / Core Ultra 9 machine unless marked as a published figure.
Landscape researched 2026-10-10.

## Picks

| Slot | Pick now | Challenger(s) | Status |
|---|---|---|---|
| STT (final) | **edda-v0.2** (Whisper turbo FT, Apache 2.0) | Brage v1 (gated), Røst-v3 | Measured, see below |
| STT (fast/partials) | **ekko-v1-tiny** (Parakeet 110M, sherpa-onnx) | — | Measured; licence is custom, check before shipping |
| VAD | Silero VAD v6 | TEN VAD | Not measured yet |
| End of turn | Pipecat Smart Turn v3.2 (Danish listed) | silence timeout | Not measured yet |
| LLM | **Gemma 4 12B** (decided 2026-10-10; Tau2 69 vs E4B's 42) | E4B as fast router, Qwen3.5-9B (Tau2 79, Danish unproven) | v0.3 bake-off on our tool traces |
| LLM runtime | llama.cpp `llama-server` (child process) | mistral.rs embedded (Rust, needs own CUDA build on Windows) | — |
| TTS (reply voice) | **English** for now ([0011](adr/0011-reply-language.md)): Pocket TTS (CPU, streaming) vs Kokoro (CUDA) | Danish voices below | Blind listening test pending |
| Embedder | harrier-oss-v1-270m (MIT, MMTEB v2 66.5) | EmbeddingGemma 2 (Apache, 61.4, multimodal) | v0.2 bake-off |
| Reranker | bge-reranker-v2-m3 | Qwen3-Reranker | v0.2 |
| Vector store | SQLite + FTS5 + sqlite-vec | Tantivy for Danish BM25, LanceDB | Decided ([0007](adr/0007-memory-model.md)) |
| AEC | **Windows comms AEC** with speakers as reference: supported on all 3 mics (probe 2026-10-10) | WebRTC AEC3 + DeepFilterNet | Echo reduction to be measured (`aec_test --play`) |

Checked and rejected for now: KittenTTS 2 (no Danish), Kokoro for Danish (no Danish), full-duplex
speech-to-speech models (none with usable Danish locally), LiveKit turn detector (no Danish),
generic multilingual ASR (Whisper large-v3, Parakeet v3, Voxtral, Qwen3-ASR: 20–30 % WER on
Danish per the RyeAI leaderboard).

## STT — FLEURS da_dk test, 60 clips (mean 10.5 s), 2026-10-10

| Model | Runtime | WER | CER | p50 | p95 |
|---|---|---|---|---|---|
| edda-v0.2 | transformers fp16 CUDA | **7.3 %** | 3.2 % | 413 ms | 548 ms |
| edda-v0.2 | CTranslate2 int8_float16 CUDA | 7.5 % | 3.3 % | 238 ms | 307 ms |
| edda-v0.2 | CTranslate2 fp16 CUDA | 7.8 % | 3.5 % | 242 ms | 307 ms |
| Røst-v3 whisper 1.5b | transformers fp16 CUDA | 9.4 % | 3.7 % | 1106 ms | 1586 ms |
| ekko-v1-tiny | sherpa-onnx int8 **CPU** | 10.7 % | 4.3 % | **127 ms** | 231 ms |

Read: edda is the most accurate and ~4× faster than v1's Røst-v3. ekko is nearly as good, runs
on CPU in a sherpa-onnx runtime the Rust engine can use directly, and its cost scales with clip
length (Whisper always encodes a padded 30 s window). FLEURS is clean read speech; the decision
stands only after CoRal conversation and Mikkel's own voice through the speakers.

## TTS Danish — 10 typical Grace sentences, 2026-10-10

`ttfa` = time until the first sentence's audio is ready. Non-streaming backends equal total
synthesis time for the sentence, which is what the user waits for when replies stream per
sentence. p95 includes the first-call warm-up.

| Voice | Runtime | ttfa p50 | ttfa p95 | RTF p50 |
|---|---|---|---|---|
| Piper da_DK-talesyntese-medium | ONNX CPU | **90 ms** | 737 ms | 0.03 |
| Supertonic 3 (F1, 8 steps) | ONNX CPU 4 threads | 844 ms | 1144 ms | 0.20 |
| Supertonic 3 (M1, 8 steps) | ONNX CPU 4 threads | 854 ms | 1227 ms | 0.21 |
| Røst-v3 Chatterbox 350m (Turbo) | PyTorch CUDA, non-streaming | 2029 ms | 2967 ms* | 0.44 |
| Røst-v3 Chatterbox 500m (Multilingual) | PyTorch CUDA, non-streaming | 3285 ms | 4612 ms | 0.89 |

\* 350m's first call took 17.6 s (CUDA warm-up); p95 shown excludes it.

Read: Piper is the only voice fast enough for a first sentence today, and the robotic one. The
Røst voices are the only ones with native-listener quality scores (MOS 4.23 / 4.01, published),
but at 2–3 s per sentence they need streaming or an ONNX/compiled path before they can lead a
reply. A Supertonic 3 run with 4 steps / 8 threads overlapped the Chatterbox download and was
discarded as noisy.

## TTS English — 10 typical Grace sentences, 2026-10-10

Grace may answer in English while Mikkel speaks Danish ([0011](adr/0011-reply-language.md)).

| Voice | Runtime | ttfa p50 | ttfa p95 | RTF p50 | Licence |
|---|---|---|---|---|---|
| **Pocket TTS** (Kyutai, alba) | torch **CPU**, streaming | **94 ms** | 232 ms | 0.33 | CC-BY-4.0 |
| Kokoro-82M v1.0 (af_heart) | sherpa-onnx **CUDA** | 250 ms | 388 ms* | 0.08 | Apache 2.0 |
| Kokoro-82M v1.0 (af_heart) | sherpa-onnx CPU 4 threads | 711 ms | 1158 ms | 0.24 | Apache 2.0 |
| Kokoro-82M v1.0 (am_michael) | sherpa-onnx CPU 4 threads | 639 ms | 1604 ms | 0.27 | Apache 2.0 |
| Supertonic 3 (F1, en) | ONNX CPU 4 threads | 1081 ms | 2199 ms | 0.27 | OpenRAIL-M |
| Chatterbox Turbo (en) | PyTorch CUDA, non-streaming | 1405 ms | 3533 ms | 0.38 | MIT |

\* Kokoro CUDA's first call took 54 s (CUDA/cuDNN warm-up); the engine must warm it at startup.

Read: Pocket TTS streams its first audio in ~90 ms using only the CPU, which keeps the GPU for the
LLM. Kokoro on CUDA is the fallback if Pocket's quality loses the listening test. Both are far
inside the latency budget; the choice is quality, by ear.

## LLM — Gemma 4 12B on llama-server, 10 Danish requests × 2, 2026-10-10

Requests use a Grace-style system prompt (~490 tokens) plus four tool definitions; the server
reuses that prefix from its KV cache, so each request prefills only the ~20–30 new tokens.
"First piece" is when the sentence splitter releases the first speakable chunk of a direct
answer: what TTS waits for.

| Config | Tool choice | TTFT p50 | TTFT p95 | First piece p50 | First piece p95 | Decode | VRAM |
|---|---|---|---|---|---|---|---|
| QAT Q4_0, c 32K, np 2, q8 KV | 20/20 | 195 ms | 320 ms | 395 ms | 649 ms | 49 tok/s | ~7.9 GB |
| + `--swa-full`, c 32K | 10/10 | 53 ms | 260 ms | 235 ms | 389 ms | 51 tok/s | ~13 GB |
| **+ `--swa-full`, c 16K** (pick) | **20/20** | **53 ms** | 243 ms | **239 ms** | 419 ms | 49 tok/s | **~9.9 GB** |

Without `--swa-full`, llama.cpp disables cache reuse for Gemma's sliding-window layers and pays
a fixed ~180 ms prefill on every request. Pick: llama.cpp b11539 CUDA 13.4,
`-ngl 99 -c 16384 -np 2 -fa on -ctk q8_0 -ctv q8_0 --reasoning off --swa-full`.
Decode at ~50 tok/s is the next lever (MTP speculative decoding, issue #26).

One finding for the harness: asked "who won the Tour de France this year?", the model searched
for "2024": it doesn't know today's date. The system prompt must carry the current date.

Quality is judged by ear: `grace-bench listen bench-results/tts` (Danish) and `grace-bench listen bench-results/en/tts` (English) build blind A/B pages.
