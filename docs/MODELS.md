# Models: current picks and bake-off results

Living register for [0009](adr/0009-models-by-bake-off.md). Numbers are from `grace-bench` on
the RTX 5090 Laptop (24 GB) / Core Ultra 9 machine unless marked as a published figure.
Landscape researched 2026-10-10.

## Picks

| Slot | Pick now | Challenger(s) | Status |
|---|---|---|---|
| STT (final) | **edda-v0.2** (Whisper turbo FT, Apache 2.0) | Brage v1 (gated), Røst-v3 | Measured, see below |
| STT (fast/partials) | **ekko-v1-tiny** (Parakeet 110M, sherpa-onnx) | — | Measured; licence is custom, check before shipping |
| VAD | **Silero VAD v6.2.3** (ONNX, 1 CPU thread, ~0.1 ms per 32 ms frame) | TEN VAD | Speech found in 60/60 FLEURS clips |
| End of turn | **Smart Turn v3.2 fp32** (CPU, ~43 ms) + silence timeout | int8 build rejected (unstable near 0.5) | 70 % on our harsh complete-vs-cut FLEURS test; published Danish 92.8 % |
| LLM | **Gemma 4 12B** (decided 2026-10-10; Tau2 69 vs E4B's 42) | E4B as fast router, Qwen3.5-9B (Tau2 79, Danish unproven) | v0.3 bake-off on our tool traces |
| LLM runtime | llama.cpp `llama-server` (child process) | mistral.rs embedded (Rust, needs own CUDA build on Windows) | — |
| TTS (reply voice) | **English** for now ([0011](adr/0011-reply-language.md)): Pocket TTS (CPU, streaming) vs Kokoro (CUDA) | Danish voices below | Blind listening test pending |
| Embedder | **harrier-oss-v1-270m** (MIT, CPU, 51 ms/query) | harrier-0.6b (better top-3, 2× slower), EmbeddingGemma 2 | Measured, see below |
| Reranker | bge-reranker-v2-m3 | Qwen3-Reranker | v0.2 |
| Vector store | SQLite + FTS5 + sqlite-vec | Tantivy for Danish BM25, LanceDB | Decided ([0007](adr/0007-memory-model.md)) |
| AEC | **Own WebRTC AEC3** (pure-Rust `aec3`) with loopback reference | Windows comms AEC: supported by the driver but only **~4 dB** on the Bluetooth Z337 | Calibration + tuning next (aec-lab) |

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

## Echo + VAD acceptance — Z337 (Bluetooth) speakers, laptop mic array, 2026-10-11

Calibration: playback→mic 110 ms (spread 13 ms). Recording: 17 s of Grace's (Kokoro) voice
through the speakers; "speech" = seconds Silero VAD + gate reports while only Grace is talking.

| Signal | False speech | Notes |
|---|---|---|
| Raw mic | 15.7 s | VAD hears all of Grace |
| Windows communications AEC | 15.2 s | no practical help on Bluetooth |
| Our AEC3 (+NS), first pass | 3.7 s | still converging on the first sentences |
| Our AEC3, adapted (`--warm`) | 0.3–0.4 s | |
| **Our AEC3, adapted + talking boost** | **0.0 s** | target met on this recording |

Double-talk (a FLEURS Danish speaker mixed in while Grace talks, adapted AEC + boost): talker at
the echo's level detected (starts ~1 s late); 6 dB quieter, patchy; 12 dB quieter (below room
noise), missed. AEC3's suppressor dampens near-end speech during double-talk: tune it, and
confirm with Mikkel's real voice over playback.

## End of turn — Smart Turn v3.2, 2026-10-11

Our Rust features match Hugging Face's `WhisperFeatureExtractor` to 1.7e-5; Rust and Python
predictions on the same files are identical. The int8 `-cpu` model gives 0.26–0.55 on one
borderline clip depending only on ONNX Runtime optimization level and threads; the fp32 model
gives 0.1004 under every setting, so fp32 it is (~43 ms per prediction, 2 threads).

Test: 60 FLEURS Danish clips, each complete and cut off at a random 40–70 % point
(`bench/make_turn_fixtures.py`). At threshold 0.5: 49/60 complete and 35/60 cut-off recognised,
70 %. Random cuts often land on clause boundaries where a pause is natural, so this undersells
real use; the published Danish figure is 92.8 %. The silence timeout in `TurnMachine` catches
"complete" turns the model misses (cost: up to 1.6 s extra wait). Premature "complete" verdicts
on cut-offs (25/60) are the risk: Grace answering too early. Measure on Mikkel's real speech.

## Embedder — Danish question → English fact, 2026-10-11

`bench/embed_recall.py` on `fixtures/memory/recall.jsonl`: 40 English facts of the kind Grace
stores plus 20 distractors; 40 Danish spoken-style questions and 3 English ones, one target each.
CPU, 4 threads.

| Model | Dim | Recall@1 | Recall@3 | MRR | ms/query |
|---|---|---|---|---|---|
| **harrier-oss-v1-270m** | 640 | **0.81** | 0.88 | 0.86 | 51 |
| harrier-oss-v1-0.6b | 1024 | 0.84 | **0.93** | **0.89** | 113 |
| EmbeddingGemma 2 | 768 | 0.77 | 0.81 | 0.80 | 51 |
| multilingual-e5-small | 384 | 0.74 | 0.86 | 0.81 | 8 |

Pick harrier-270m: nearly the 0.6b's accuracy at half the latency (it sits on the voice path).
Typical misses are indirect phrasings ("hvem er du opkaldt efter" → "named after Grace Hopper");
hybrid keyword search and a reranker are the next levers.

Quality is judged by ear: `grace-bench listen bench-results/tts` (Danish) and `grace-bench listen bench-results/en/tts` (English) build blind A/B pages.
