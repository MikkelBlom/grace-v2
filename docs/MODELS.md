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

## TTS Danish — Røst v3 Chatterbox, streamed, 30 sentences × 2, 2026-10-11

`bench/synth/roest_bakeoff.py` (venv `bench/.venv-roest`, torch 2.11 cu128 from the global
Python). `fast` = `bench/synth/roest_fast.py`: same weights and sampling, but static KV cache,
every T3 decode step and the S3Gen chunk windows as CUDA graphs (kernel launches cost ~20–50 µs
each on this Windows box, so stock is launch-bound), bf16, sentence streamed with a 16-token
(0.64 s) first chunk. Stability = edda-v0.2 transcript vs text (CER, numbers normalised).
Voice: the repos' `conds.pt`, which is Resemble AI's default speaker, not CoRal's Mic/Nic (those
runs follow below the table).
**The GPU was 50–98 % busy with other work during every run**, so absolute times are inflated.

| Config | ttfa p50 | ttfa p95 | RTF p50 | CER | clips > 20 % CER | VRAM |
|---|---|---|---|---|---|---|
| 350m stock `generate()` | 6864 ms* | 11432 ms* | 1.72 | 4.5 % | 3/60 | 3.3 GB |
| **350m fast, streamed** | **236 ms** | 295 ms | 0.29 | 5.7 % | 8/60 | 4.4 GB |
| 350m fast, whole utterance | – | – | 0.42 | 3.8 % | 4/60 | 4.5 GB |
| 500m stock `generate()` | 11163 ms* | 21288 ms* | 3.14 | **2.3 %** | 0/60 | 3.6 GB |
| 500m fast, streamed, 2-step S3Gen | 300 ms | 361 ms | 0.37 | 3.3 % | 3/60 | 5.6 GB |
| 500m fast, streamed, 2-step, no CFG | 324 ms | 375 ms | 0.38 | 2.6 % | 0/60 | 5.3 GB |
| 500m fast, whole utterance, 10-step S3Gen | – | – | 1.29 | 3.3 % | 3/60 | 5.5 GB |
| 500m fast, whole utterance, 2-step S3Gen | – | – | 0.69 | 3.3 % | 2/60 | 5.5 GB |

\* non-streaming: ttfa = whole sentence. Same configs on the quieter GPU of 2026-10-10: 2.0 s / 3.3 s.

Single-round runs show how much the shared GPU moves the result: 350m streamed 163 ms (p95 177)
with the GPU 62 % busy, 144 ms with a 12-token first chunk; 500m 200 ms. **With llama-server
decoding at the same time** (what happens while Grace speaks her first sentence): 350m 304 ms
(p95 316); with a 12-token first chunk 271 ms but 27/30 sentences had audible gaps (1.7 s total).
Stage floors (`roest_parts.py`, best of many repeats): 350m prefill ~15 ms + 3.8–4.4 ms/token +
first S3Gen window 37–59 ms ≈ **136 ms**; 500m 17–31 ms + 7.7–9 ms/token (CFG doubles the rows,
30 layers) + ~50 ms ≈ **242 ms**. The 500m's own 10-step S3Gen costs ~480 ms per window; it only
works live with the 350m's 2-step "meanflow" decoder (same S3 tokens; same CER here). Shorter voice
prompt for S3Gen (4 s instead of 10 s) saves ~20 ms per window. torch.compile was not tried
(needs Triton, not installed); the CUDA graphs remove the launch overhead it would target.
fp16 vs bf16 not separated (same bytes moved; contention noise is larger than the difference).

Read: the fast path makes Røst streamable (stock is 10–40× too slow), but first audio is
~140–170 ms only when the GPU is otherwise quiet and ~270–300 ms while the LLM decodes — over the
≲ 250 ms budget, with gaps unless chunks grow gradually. The 500m is the clearer speaker
(lower CER) and ~40–60 ms slower to first audio. Neither has been judged by ear yet:
`bench-results/tts-da/listen/` (5 clips each: 350m, 500m, 500m stock reference).

Licence (both repos, `LICENSE`, read 2026-10-11): Alexandra Institute "Software – license to use",
AI Pubs OpenRAIL-M text plus Danish law and arbitration in Aarhus. Commercial use allowed, but
the Attachment A use restrictions travel with the model, its derivatives and its output; among
them: no synthesising or modifying a natural person's voice without that person's prior informed
consent (4b); no autonomous audio interaction with a person unless they were told beforehand it is
not a human (4c); no medical advice without accreditation (6); the licensor may restrict use
remotely or by update (7). Watermarking is not a licence term: the Perth watermark is a
`chatterbox-tts` default (`roest_fast` skips it for live use, `tts_da.py` keeps it).
Weights: only T3 is fine-tuned; `s3gen`, `ve` and `conds.pt` are byte-identical to Resemble AI's
MIT releases. CoRal's Mic and Nic (the voices behind MOS 4.01/4.23) are not shipped; using them
needs reference clips from `CoRal-project/coral-tts` (CC0, two professional speakers, made for
speech synthesis; not the OpenRAIL-D CoRal ASR set).

### With CoRal's Mic and Nic, quiet GPU (4–11 % busy), 30 sentences, 2026-10-11

Mikkel approved CoRal TTS prompts: one ~9 s CC0 clip per speaker
(`fixtures/corpora/voices/coral-tts/`, in the voice manifest as `coral-mic` / `coral-nic`).

| Config | ttfa p50 | ttfa p95 | RTF p50 | CER | clips > 20 % CER | gaps |
|---|---|---|---|---|---|---|
| 350m fast, Mic | 111 ms | 117 ms | 0.15 | 8.4 % | 2/30 | 0 |
| 350m fast, Nic | 123 ms | 132 ms | 0.16 | 12.3 % | 4/30 | 0 |
| 350m fast, built-in voice | 120 ms | 133 ms | 0.16 | – | – | 0 |
| **500m fast (2-step S3Gen), Mic** | **142 ms** | 153 ms | 0.20 | **1.3 %** | 0/30 | 0 |
| **500m fast (2-step S3Gen), Nic** | **158 ms** | 169 ms | 0.21 | **0.8 %** | 0/30 | 0 |

On a quiet GPU both models are inside the ≲ 250 ms budget. With Mic/Nic the 500m is far more
accurate than the 350m (which mispronounces more with these prompts; the card's temp 0.7 /
top_k 600 did not help: Mic 10.5 %, Nic 13.5 %). One 500m Mic run with llama-server decoding *and*
another job holding the GPU at 95 % gave 538 ms and gaps in 27/30 sentences: contention, not the
model, is the open risk, so TTS and LLM decode must be scheduled/measured together in a live test.
Listening clips: `bench-results/tts-da/listen/roest-{350m,500m}-{mic,nic}_*.wav`.

### Female voices for Røst (Mikkel: female only, young and warm), 2026-10-11

Røst rule 4b needs the speaker's prior informed consent, so prompt sources were checked:
- **Pocket TTS "azelma" / "eve"**: VCTK speakers p303 / p361 (CC BY 4.0). Recorded for a
  voice-cloning *research* corpus. No published consent to synthesis of their voices: **not used**.
- Expresso, EARS (other Pocket/Kyutai voices): CC BY-NC, no consent statement: not used.
- **Kyutai voice donations** (`kyutai/tts-voices/voice-donations`, 228 voices, CC0): donors ticked
  consent boxes accepting that TTS models may generate speech in their voice (Terms of Use v1). They are
  usable, but have no gender labels, and pitch is not enough to rule out male voices. Fetched by
  `bench/synth/kyutai_voices.py`, not rendered yet.
- **A volunteer friend of Mikkel's** (private recording, consent given to Mikkel, local use only).
  Two ~10 s prompts were cut from it. 500m fast path, quiet GPU: first audio p50 158 ms / p95 167 ms
  (10 sentences). CER on the 5 listening sentences: prompt B 0.3 %, prompt A 5.3 %.
  Clips are in `bench-results/tts-da/listen-female/` (never commit).

## Synthetic Danish speech — `bench/synth/tts_da.py`, 2026-10-11

For other tracks' training/test data: `synthesize(text, voice_id) -> (float32, 16000)` and a CLI
(JSONL in, 16 kHz WAVs plus a manifest out, `--check` re-synthesises clips edda can't transcribe).
Voices: 24 cloning prompts (12 F / 12 M, ages 18–61, 7 dialect regions, 10–19 s, SNR 33–45 dB) from
NST Danish (CC0, Språkbanken; read from the `alexandrainst/nst-da` mirror), built by
`build_voice_bank.py` into `fixtures/corpora/voices/` with source and licence per voice. Common
Voice da was skipped (Mozilla Data Collective needs an account); CoRal audio is never used.

Engine: **Resemble AI's multilingual Chatterbox (MIT) + the 2-step meanflow S3Gen**, not Røst:
cloning NST speakers would breach Røst's rule 4b (they never consented to voice synthesis).
Bake-off on the same 30 sentences: CER 2.7 % (2/60 > 20 %), RTF 0.34 on the busy GPU (10-step
S3Gen: RTF 0.66, CER 5.7 % with another voice). 24/24 bank voices came out intelligible (mean CER
1.2 %, max 9 %), and a speaker-embedding check put every clip nearest its own prompt (24/24; cosine
0.84 to own prompt vs 0.75 to the nearest other).

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

## STT in simulated rooms — grace-sim pilot, CoRal v3 conversation test, 2026-10-11

40 scenes per set; people from CoRal (24 test speakers), rooms simulated (RT60 0.25-0.6 s),
Grace's echo calibrated to real Z337 recordings; cut at the true speech edges. Pilot noise was a
synthetic fallback (DEMAND/ESC-50 still downloading), so far-field numbers are pessimistic.

| edda-v0.2 int8, greedy + fallback (prod) | raw mic | AEC3 linear | AEC3 full chain |
|---|---|---|---|
| clean (corpus as recorded) | 14.3 % | — | — |
| far (1.5-3 m, SNR 10-20 dB) | 43.8 % | 46.0 % | 48.8 % |
| double-talk (Grace talking, 0.8-2 m) | 41.3 % | 40.3 % | 47.4 % |

Decoding on the linear signal: fp16 beam 5, temperature 0 → clean 16.6 / far 42.9 / double-talk
38.6 %; int8 greedy, temperature 0 → 15.0 / 46.3 / 40.1 %. Within noise at n = 40; latencies
were inflated by a concurrent benchmark. Decision: [ADR 0014](adr/0014-stt-reads-linear-aec.md).

## Memory — grace-memory-bench, 40 Danish multi-session scenarios, 2026-10-11

Gemma 4 12B extractor (temperature 0.2), probes answered at temperature 0. 246 checks.

| Category | v0.2 | v0.2.5 ([ADR 0015](adr/0015-memory-facts-and-commitments.md)) |
|---|---|---|
| **all checks** | **206 (84 %)** | **217 (88 %)** |
| self_echo | 23/28 | 28/28 |
| abstention | 25/28 | 28/28 |
| commitment | 32/49 | 39/49 |
| proactive reminder raised | 6/10 | 9/10 |
| commitment closed when done | 2/7 | 4/7 |
| multi_session | 67/78 | 73/78 |
| temporal | 58/80 | 65/80 |
| knowledge_update | 62/71 | 64/71 |
| same_session_correction | 34/37 | 32/37 |
| interrupted | 24/30 | 24/30 |
| dialect | 29/35 | 28/35 |

Remaining failures: renames/updates said mid-session in an interrupted turn, Danish number
words ("femoghalvfems" stored as ninety), commitments not closed after being acted on. Run-to-run
noise of the extractor is a few checks; compare categories, not single checks.

## Non-speech → STT — 100 ESC-50 room-sound scenes through grace-sim, 2026-10-11

VAD (full chain) triggered on 1/100 scenes (2.4 per hour). Fed directly to STT anyway:
edda-v0.2 wrote text on 100/100 (avg_logprob filter caught 50); ekko-v1-tiny on 8/100, single
words. The ekko cross-check keeps 40/40 clean, 36/40 distant, 39/40 talked-over pilot utterances.
Decision: [ADR 0016](adr/0016-hallucination-check-by-transducer.md). Under a busy GPU (other
tracks training) edda took 1.3-4 s per clip, ekko 60-200 ms on CPU; edda latency must be
re-measured on an idle GPU.

Memory run d (unknown-id updates kept as adds, number words in the duplicate check, commitments
plus facts): 216/246 vs 217 (within noise); interrupted 24 → 28 of 30.

Memory run e (end-of-session reconciliation pass on top of d): 214/246 — no gain; probes 55 → 50
and self_echo 28 → 23 (the pass sees Grace's lines in the transcript). Kept off; code stays
behind `grace-memory-bench --reconcile`. Run-to-run noise between c and d: ±1-4 per category.

## STT in simulated rooms — full sets, 150 utterances each, DEMAND room noise, 2026-10-11

| WER, oracle cuts | raw / prod | linear / prod | full chain / prod | linear / fp16 beam 5 | linear / ekko |
|---|---|---|---|---|---|
| near 0.4-0.8 m, SNR 25-35 | 20.8 | 21.9 | 22.2 | 21.4 | 28.2 |
| far 1.5-3 m, SNR 10-20 | 26.5 | 27.2 | 28.0 | 26.9 | 38.9 |
| double-talk 0.8-2 m | 36.6 | 32.1 | 43.5 | 30.1 | 44.5 |

Real speech wrongly dropped as hallucination: ekko check 1 / 2 / 8 of 150; old avg_logprob
filter 7 / 16 / 21. grace-sim detection: near 149/150, far 143/150, double-talk 129/150
(onset 1.4 s); media playback 0 triggers in 60 scenes, Grace's echo 1 in 60.

## End of turn — Smart Turn v3.2 fine-tuned on Danish ("da1"), 2026-10-11

Full fine-tune on CoRal v3 train (76.5k clips: complete = segment followed by the other
speaker; incomplete = cut at a word gap via wav2vec2 alignment, or a mid-turn pause on øh/og)
plus a third of the v3.2 multilingual train set, then 70 % blended with the original weights
(blend chosen on CoRal validation speakers and a held-out v3.2 slice). Same ONNX contract;
Rust output matches Python (max diff 5e-5); 35 ms vs 33 ms on CPU. Scripts: bench/smart_turn/.

| Test | v3.2 | **da1 (live)** | unblended |
|---|---|---|---|
| CoRal conversation test, 3,681 clips, 24 unseen speakers: accuracy | 60.4 % | **74.7 %** | 77.0 % |
| — false "turn ended" (share of unfinished clips called complete) | 39.2 % | **22.4 %** | 19.8 % |
| Harsh FLEURS cut test (120): accuracy / false "turn ended" | 70.0 / 41.7 % | **84.2 / 16.7 %** | 85.0 / 16.7 % |
| v3.2 test, Danish (779, synthetic): accuracy | 93.58 % | 93.32 % | 92.81 % |
| v3.2 test, all 23 languages (31,527): accuracy | 93.71 % | 93.65 % | 93.18 % |

Baseline reproduces Daily's published v3.2 numbers. CoRal labels are heuristic (noisy). The
model is derived from CoRal (OpenRAIL-D): if ever shared, its licence and use restrictions go
with it. Not yet done: synthetic Danish fillers (TTS too slow on the busy machine), threshold
and silence-timeout retuning, live check.

## Code-switching — 40 Danish sentences with English tech words, 24 synthetic NST voices, 2026-10-11

`fixtures/stt/codeswitch.da.jsonl` voiced by `bench/synth/tts_da.py` (MIT Chatterbox). edda-v0.2
int8: no hint 24.1 % WER, vocabulary `initial_prompt` 24.3 %, `hotwords` 24.6 %: edda ignores
vocabulary hints (and the prompt changes nothing on CoRal near 21.9 → 21.8 %, double-talk
32.1 → 31.8 %). Typical errors: itslearning → "islaring", tools → "Thols", push → "pudse",
Docta → "doktor". Real fix: LoRA fine-tune on synthetic code-switched speech (FUTURE_IDEAS).

## Smart Turn threshold — da1 on CoRal validation (993 clips), 2026-10-11

| threshold | finished turns noticed | unfinished cut off |
|---|---|---|
| v3.2 @ 0.5 (old) | 63.7 % | 40.0 % |
| da1 @ 0.5 | 75.6 % | 23.6 % |
| **da1 @ 0.3 (live)** | **83.7 %** | 31.2 % |
| da1 @ 0.1 | 91.6 % | 47.2 % |

Live run 3 with da1 @ 0.5 waited 1.6-3 s on complete questions (scores 0.06-0.41). Unfinished
threshold lowered 0.1 → 0.05 (fewer 3 s waits: 8 → 5 % of finished turns).

## Turn-taking policy — grace-turn-sim, 250 CoRal turns, 2026-10-20

`grace-turn-sim` drives the real `TurnMachine` in virtual time: real frontend (AEC3, Silero,
gate), Smart Turn da1 and edda STT (results cached per span), LLM + TTS as a constant 620 ms
(live run 4 medians: recall 94 + commit → first audio 522). Scenes (`bench/sim/make_turns.py`):
150 "multi" turns, 2-3 consecutive segments by one CoRal test speaker joined by a 0.35-3 s pause
(median 0.84 s), and 100 "single" finished turns; clip mic, DEMAND bed 25-35 dB down; the person
doesn't react to Grace (headset). Latency = true end of speech → Grace's first audio queued
(Bluetooth adds ~150 ms live). Decision: [ADR 0018](adr/0018-recoverable-turn-ends.md).

| policy | set | answered in halves | ...Grace heard answering half | jumped in, then merged (heard) | p50 | p90 |
|---|---|---|---|---|---|---|
| before (live runs 1-4) | multi | 63 % | 44 % | — | 1.67 s | 3.66 s |
| before | single | 3 % | 2 % | — | 1.44 s | 3.60 s |
| new thresholds only (no merge, no prepare) | multi | 75 % | 59 % | — | 1.43 s | 2.67 s |
| new thresholds only | single | 3 % | 2 % | — | 1.28 s | 2.61 s |
| **after (merge + prepared replies + new thresholds)** | **multi** | **3 %** | **2 %** | 43 % (0.4 s avg) | 1.55 s | **2.07 s** |
| **after** | **single** | **1 %** | **0 %** | 3 % | **1.24 s** | **2.04 s** |

Early commits ("jumped in" before the true end): 63 % → 77 % of multi turns; 98 % of them are
now merged (31 % before Grace was audible, 43 % after up to 1.5 s of her). LLM requests per
multi turn 1.9 → 2.8 (0.7 prepared and discarded). Sweeps (bench/sim/turn_sweep.py): VAD
hangover 8 → 4 frames −160 ms p50; long wait 3 → 2 s p90 3.0 → 2.1 s; fallback 1.6 → 1.0 s
p50 −0.1 s on multi; 0.8 s / 1.5 s more heard merges (53 %) for p90 −0.1-0.4 s; threshold 0.2 or
0.5 vs 0.3 within ±0.1 s; min_commit_silence 500/700 ms no effect; merge window 2.5 s removes
the last heard splits (2 → 0 %) but would absorb real interruptions up to 2.5 s into a reply.

Merged turns are transcribed again whole: edda WER 16.4 % on the whole paused turn vs 18.7 %
for the parts' transcripts joined (150 turns, 4,238 words, oracle edges).

Sim STT took 451 ms p50 (live run 4: 298 ms; llama-server and Pocket TTS were loaded, long
turns), so absolute latencies are pessimistic; the comparison is like for like.

## Style learning — grace-style-bench, 7 scenarios × 3 seeds, 2026-10-20

Gemma 4 12B (QAT q4_0) for replies (engine settings), the style classifier and the judge; main's
`config/prompts/system.md`. "off" = today's behaviour (feedback only in the chat history),
"on" = rules learned by the classifier, rendered in the system message and restated on the
current user message ([ADR 0017](adr/0017-style-rules-and-review-queue.md)). Run `v5`.

| | off | on |
|---|---|---|
| **hard checks + judge** | **48/69** | **241/243** |
| 01 shorter: mean words before → after feedback → next session | 49 → 15 → **43** | 51 → 13 → **21** |
| 01 + 04: replies within the word cap after feedback | 9/15 | 15/15 |
| 02 fewer questions: question rate before → after → next session | 1.00 → 0.67 → **0.67** | 0.78 → 0.00 → **0.00** |
| 05 lists its rules / drops one by voice | 4/9 | 9/9 |
| 07 ten rules at once: thorough technical answer (words), judge | 195, 4/4 | 119 (≥ 40 in 3/3), 4/4 |
| classifier: rule changed on feedback turns / on other turns | — | 18/18 / 1/120 |

The remaining false rule: "nej det er forkert, Aarhus er ikke hovedstaden" (a fact correction)
became "be careful with geographical facts" in 1 of 3 seeds.

Iterations (same scenarios): v1 learned 6/120 false rules ("hold lige kæft" after a long reply →
"be concise", 3/3 seeds; a neutral line after a broken rule → a duplicate rule) → code guards
(new/changed rules need the feedback flag; hush-only evidence refused). Requiring the flag for
removals too (v2) lost 6/18 real feedback turns; now removals need only his words. Rules in the
system block only (v4): next-session question rate 0.67, same as off; restating them on the
user message (v4-remind, adopted): 0.00, and 243/243.

Memory regression (`grace-memory-bench`, same day, same llama-server): base 6420bd6 220/246,
this branch 225/246. A first version that told the fact extractor to leave talking style to the
rules scored 206: one extra sentence in its prompt cost unrelated extraction (sisters, cousins);
the extractor prompt is unchanged now. Run-to-run noise ±1-4 per category; the old 216 figure
above is from 2026-10-11.

## Conversation style — grace-chat-bench, 13 scenarios × 3 seeds, 2026-10-20

Gemma 4 12B at the engine's settings (temperature 0.6, seeded), real `system_message` path, fake
date. Multi-turn scenarios use a simulated user (Gemma playing a fictional Danish persona,
`fixtures/chat/persona.md`). Hard checks in code; judge = the same Gemma at temperature 0 with
`docs/conversation-style.md` as rubric. Score = mean of hard and judge pass rates. Scenarios
from live run 4's failures (#36, #37).

| | Baseline prompt | Tuned prompt (#36) | Tuned, repeat run |
|---|---|---|---|
| **score** | **0.793** | **0.975** | **0.980** |
| hard checks | 538/625 | 604/616 | 605/615 |
| judge criteria | 122/168 | 163/168 | 164/168 |
| quiet when told ("Okay." or nothing) | 3/21 | 21/21 | 21/21 |
| length caps | 84/126 | 123/126 | 120/126 |
| banned phrases | 127/147 | 142/147 | 144/147 |
| one question at a time | 63/69 | 68/69 | 69/69 |
| speakable (no markdown / model tokens) | 146/147 | 147/147 | 147/147 |
| run time | 676 s | 445 s | — |

Per scenario, judge: interview with empty memory 9/21 → 20/21, be quiet 3/9 → 9/9, dictation
6/12 → 12/12, fillers 5/9 → 9/9, no-tools 9/12 → 9/12 (unchanged). The first baseline run, on
an earlier harness (cruder repeat check), scored 0.729.

How much to trust it: run-to-run noise is about ±0.01 on the tuned prompt (seeded, but
llama-server isn't bit-exact across runs). The judge is Grace's own model and is lenient: it
passed a reply that was only leaked `<|channel>` tokens as "silent", so the hard checks carry the
safety-relevant rules. Leftovers: "That sounds like…" as a reflex (5/147), asking for studies
already in memory (2/9 runs), "haha" still re-asks the question in 1/3 seeds.

### Candidate styles for the Jarvis-like persona, 13 scenarios × 2 seeds, 2026-10-20

Same rules, only the personality paragraph swapped (`fixtures/chat/styles/`). Harness now also
judges "not trying too hard" and hard-checks exclamation marks, so these scores don't compare
with the table above. Judge rubric was still the "sharp friend" spec, which marks "sir" and
formality down: butler's lower judge score is partly that rubric, not Mikkel's taste.

| Style | score | hard | judge |
|---|---|---|---|
| current (tuned #36 prompt) | 0.979 | 505/508 | 133/138 |
| friend (warm-sarcastic) | 0.973 | 500/513 | 134/138 |
| minimal | 0.967 | 497/508 | 132/138 |
| butler (Jarvis-like, occasional "sir") | 0.942 | 506/513 | 124/138 |

All three new styles regressed on asking what memory already holds (4/32 vs 0/32) and on "That
sounds like…"; minimal leaked `<|channel>` tokens on fillers twice. The pick is Mikkel's
(blind page, `bench/style_pick.py`); fold the winner in, then fix those regressions.
Also seen in the pick set: "sytten gange treogtyve" answered 510 and 51 (correct: 391) in three
of four styles.

### Reply language, 2026-10-20

New scenario 14 (casual Danish openers, "kan du ikke bare svare på dansk") and a global hard
check (two or more Danish function words = not English). Reported by the style-learning
session. English replies in scenario 14, 3 seeds × 5 turns:

| Prompt | English | judge |
|---|---|---|
| main's system.md | 0/15 | 0/15 |
| tuned #36 prompt | 11/15 | 7/15 |
| + explicit "Reply language" paragraph | 15/15 | 15/15 |

The other 13 scenarios were already 54/54 English. Full bench with the language paragraph:
0.949-0.965 over two runs (14 scenarios × 3 seeds; judge varies by ±7 of 222 between runs and
once cited exclamation marks that the hard check found 0 of). Asking for facts memory already
holds (scenario 02) is 43-45/48 on both the old and new prompt: a standing weakness, not a
regression; Gemma reads past the memory block at the end of the prompt.

### Mikkel's own replies as ground truth, 14 scenarios × 3 seeds, 2026-10-20

Round 1 of the blind pick failed (he picked none on 8 of 20); his own replies are now the
few-shot block in system.md and the judge's reference. Judge criteria were rewritten around them
("short, casual, like someone who knows him"; one "!" and "haha" allowed), so scores don't
compare with the tables above.

| Prompt | score | hard | judge | "That sounds like" | small talk ends in "?" ≤ 50 % |
|---|---|---|---|---|---|
| his examples folded in | 0.888 | 959/985 | 178/222 | 15 | 1/3 |
| + "never start with That sounds like", fewer follow-up questions | 0.898 | 961/985 | 182/222 | 11 | 0/3 |
| + round-2 picks (16 more of his examples) and scenario 15 (memory-aware opinions) | 0.909 | 1027/1049 | 199/237 | 12 | 1/3 |

Stable: quiet 21/21, English 162/162, dictation, fillers, greetings, technical, no-tools.
Weak under the new rubric: small talk (judge 7/18: she interviews him), interview with memory
(42/48 known-fact questions avoided). Round-2 pick page samples also showed "I cannot set
reminders" (wrong: memory commitments do that) in 1 of 2 samples despite the new rule, and
"That is a classic move". Not tuned further: his round-2 picks are the better signal.

Round 2 of the pick: plain 4, tease 7, Grace 5, none 0 (round 1: none on 8 of 20). Scenario 15
(memory says he hates Python and likes C sharp): hard 63/63, judge 15/15. Small talk is still
the weak spot (judge 7/18: she keeps asking questions); "That sounds like" persists (12 of 171).

### Principles instead of a pair list, held-out lines, 2026-10-20

Mikkel objected to 24 exact example pairs in system.md. The picks were distilled into general
rules ("Your manner") plus 7 illustrations marked "never reuse their wording". Compared on 22
fresh, hand-written everyday lines (`fixtures/chat/heldout/`, none in the prompt, bench or pick
sets), 3 seeds = 66 replies each. Reuse = replies containing an example sentence of 4+ words
verbatim (`bench/example_reuse.py`); openers from `bench/reaction_openers.py`.

| Prompt | held-out score | hard | judge | verbatim reuse | top opener |
|---|---|---|---|---|---|
| pair list (52870dd) | 0.965 | 391/399 | 208/219 | 9/66 ("Nice. Congratulations, Mikkel!", "Good morning, Mikkel", "What is the issue?") | "That…" 24/66 |
| principles v1 | 0.984 | 397/399 | 213/219 | 6/66 (all "Nice. Congratulations, Mikkel!") | "Oh no" 29/66 |
| v2: congratulations example removed | 0.977 | 399/399 | 209/219 | 3/66 ("What is the issue?" on a git problem) | "Oh no" 26/66 |
| **v3: reactions must fit and vary (adopted)** | **0.995** | **399/399** | **217/219** | **3/66** (same) | "Nice" 17, "Ugh" 12, "Oh no" 6 |

Full bench (15 scenarios × 3 seeds): pair list 0.909 → principles 0.929, even though the pair
list contained some bench lines word for word. Small talk is still weak (judge 6/18; every reply
ends in a question). Caveat: v2 and v3 were tuned against this held-out set, so it is no longer
held out; the next round needs new lines.


## Fixes from live run 5, 2026-10-21

**Style classifier** (`grace-style-bench`, 8 scenarios × 3 seeds, learning on). New scenario 08 =
run 5's five false rules, paraphrased (didn't listen, a story of N words, "not a comedy",
resuming after an interruption).

| Classifier | 08 checks | false rule changes in 08 | full: feedback turns learned | full: false changes | full checks |
|---|---|---|---|---|---|
| main (ADR 0017) | 17/27 | 7/24 | — | — | — |
| + prompt lines for those cases | 20/27 | 5/24 | — | — | — |
| + required `about` category, code keeps manner only | 27/27 | 0/24 | 15/18 | 0/144 | 255/270 |
| + "an update keeps the old rule's untouched parts" (adopted) | 27/27 | 0/24 | **18/18** | **0/144** | **265/270** |

Scenario 04 (update, not pile up) was 33/33 on main and 29/33 before the last line; 02 (fewer
questions) is 42-43/48 on both, so its misses are her replies, not the classifier.

**Persona prompt** (`grace-chat-bench`, 16 scenarios × 3 seeds, same day, same binary): previous
system.md 0.942 (hard 1104/1122) → honesty + no question tail + no compliance announcements
0.956 (hard 1107/1122, judge 233/252). New scenario 16 (two made-up titles, two actor questions):
invented plots 3/3 → "I don't know that one" 3/3; "who played Sean Parker" is Andrew Garfield in
3/3 on both prompts (it's Justin Timberlake), a knowledge gap, not a prompt one.
