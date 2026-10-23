# Quantization: weights (QAT Q4_0 vs alternatives), FP8/NVFP4, KV cache

Desk research 2026-10-23. Sources are linked inline; "secondary" = blog/forum, not primary.

## 1. What it is
Weights can be stored at 4-8 bits; the KV cache (attention memory) can also be stored at 8 or 4
bits. Lower bits free VRAM and, because decode is memory-bandwidth-bound, can raise tok/s, at some
quality cost. Grace today: Google's QAT Q4_0 GGUF (6.98 GB for 12B), KV `q8_0`/`q8_0`, `-fa on`,
`--swa-full`, `-c 16384 -np 2`, about 9.9 GB VRAM and about 50 tok/s
(`docs/MODELS.md` LLM section).

## 2. Evidence

### 2a. Gemma 4 12B weight formats
- Google released QAT checkpoints for Q4_0 (GGUF for llama.cpp, compressed-tensors for vLLM) and
  says QAT beats post-training quantization, but published **no numbers** for it
  ([Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)).
  Treat "QAT is near-lossless" as a vendor claim.
- Unsloth (vendor of a competing quant, so interested) measured KL divergence against BF16 on the
  QAT release ([Unsloth QAT page](https://unsloth.ai/docs/zh/mo-xing/gemma-4/qat)). 12B row:

  | | Size | Avg KLD | 99.9% KLD | Top-1 agree |
  |---|---|---|---|---|
  | Google Q4_0 | 6.98 GB | 0.507 | 14.7 | 74.1 % |
  | Unsloth UD-Q4_K_XL (from QAT weights) | 6.72 GB | 0.133 | 9.27 | 88.8 % |

  Their claim: a plain Q4_0 conversion of the QAT BF16 weights reproduces Google's own bytes only
  about 25 % of the time, and their method gets 99.96 %. Same pattern at every size (31B: 0.093 vs
  0.014). Note the absolute 12B KLD is high for both (0.13-0.5); the 12B looks unusually sensitive
  compared to 31B and E4B. Caveat: KLD against BF16 is not the same as task accuracy; QAT weights
  were trained to be good *at Q4_0*, and KLD vs the BF16 teacher may overstate the loss. A Reddit
  thread (secondary, anecdotal) mentions user reports of degradation
  ([snapshot](https://reddit.sentinel-team.org/posts/1tzxmm8/snapshots/2026-06-13T02%3A56%3A06.64152Z)).
- I found **no published Gemma 4 12B numbers** for Q4_K_M, IQ4_XS, Q5_K_M, Q6_K or Q8_0 on tool
  calling or instruction following, and none from independent evaluators of QAT Q4_0 vs Q4_K_M. The
  generic rule ("Q4_K_M keeps sensitive layers at higher precision, so beats plain Q4_0 at similar
  size") is secondary ([avenchat](https://avenchat.com/blog/gemma-4-q4-vs-q8)). Expected file sizes
  for a 12B (my arithmetic, not measured): IQ4_XS about 6.3 GB, Q4_K_M about 7.3 GB, Q5_K_M about
  8.5 GB, Q6_K about 9.8 GB, Q8_0 about 12.7 GB.
- Speed: decode is bandwidth-bound, so tok/s scales roughly inverse to file size on the same
  kernels; Q4_0 has simple, fast kernels. Expect Q8_0 about 0.55x of Q4_0 decode, Q5/Q6 about
  0.7-0.85x, Q4_K_M about 0.9-0.95x. (Expectation, not measured here.) Note Grace's measured 50
  tok/s with a 7 GB file is well under the ~100+ tok/s a pure-bandwidth estimate gives for 896 GB/s
  GDDR7, so something else (laptop power, kernel launch overhead, `-np 2`, SWA) bounds decode and
  a smaller file will give less than proportional gain. See `laptop-hardware.md`.

### 2b. FP8 / NVFP4 / MXFP4 on Blackwell
- Blackwell (sm_120, which includes the 5090 Laptop) has native FP8 and FP4 tensor cores.
- **llama.cpp NVFP4**: type `GGML_TYPE_NVFP4` exists; CUDA dp4a kernel (#20644, Mar 2026), MMQ
  (#21074), and a Blackwell-native FP4 path that reportedly landed in b8967 (Apr 29 2026)
  ([InsiderLLM, secondary](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)).
  Reported first-day RTX 5090 numbers (one Reddit post, Qwen3.6-27B NVFP4): prefill +43-68 %,
  **decode unchanged (73.7 vs 73.6 tok/s)**. The same page says NVFP4 saves only about 7 % size
  versus Q4_K_M and that quality vs Q4_K_M is "still being tested". Grace's b11539 is far newer than
  b8967, so the kernels are present; what is missing is a Gemma 4 12B NVFP4 GGUF. None found. Google
  ships Q4_0 and mobile formats, not NVFP4.
- **Is there an NVFP4 Gemma 4 12B?** Yes, for vLLM, not llama.cpp. Community/vendor
  checkpoints: [RedHatAI/gemma-4-12B-it-NVFP4](https://huggingface.co/RedHatAI/gemma-4-12B-it-NVFP4)
  (llm-compressor GPTQ, weights+activations NVFP4, "preliminary", vLLM only, no GGUF mentioned) and
  [AtomicChat/gemma-4-12B-it-NVFP4](https://huggingface.co/AtomicChat/gemma-4-12B-it-NVFP4).
  NVIDIA's own NVFP4 Gemma 4 checkpoint I found is the 31B, not the 12B. RedHat's card (vendor
  numbers, GSM8K Platinum 5-shot only): BF16 0.908; NVFP4-GPTQ 0.902 (99.3 % recovery); RTN 0.889;
  iMatrix 0.897; but byte-perplexity recovery only 89.6 % for GPTQ (PPL 4.20 vs 3.76 BF16), i.e.
  modelled language quality drops noticeably while a math benchmark barely moves.
  **No comparison with Google's QAT Q4_0 exists** (no common benchmark, and NVFP4 here is
  post-training quantization, not QAT), and nothing on tool calling or instruction following. Rough
  size: about 7-8 GB, so no VRAM win. A llama.cpp NVFP4 GGUF could be made by converting the
  BF16 weights (the convert script for compressed-tensors NVFP4 was still an open PR, #21095, per
  the InsiderLLM page), but it would be our own unvalidated artefact. Net: NVFP4 for Gemma 4 12B
  needs vLLM (WSL2) to use the published files, which `inference-engines.md` argues against.
- **MXFP4 in llama.cpp**: exists (gpt-oss uses it), but a Feb 2026 issue reported sm_120 build
  failures on MXFP4 templates ([#19662](https://github.com/ggml-org/llama.cpp/issues/19662));
  probably fixed by now (b11539 is far later) but unchecked. MXFP4 is a poor target for a model not
  trained for it.
- **FP8**: runs in vLLM/SGLang/TensorRT-LLM (not llama.cpp). 12B at FP8 is about 12 GB weights: no
  VRAM win over the current setup and slower decode than Q4 (twice the bytes per token).
- Bottom line: decode on a 12B at batch 1 is bandwidth-bound, so FP4 compute gains mostly help
  prefill. Prefill of 20-30 new tokens per turn is already about 50 ms with the cache hit.

### 2c. KV cache quantization (llama.cpp)
- Flags `-ctk/-ctv` accept f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1. A **quantized V cache
  requires flash attention** (`-fa on`); Grace already has it on.
- A KL-divergence study ([localbench](https://localbench.substack.com/p/kv-cache-quantization-benchmark),
  one author) on Gemma 4: 31B dense q8_0 KV KL 0.108; 26B-A4B MoE q8_0 KL **0.377** and q4_0 KL
  **1.088** (68 % top-1 agreement). Qwen 3.6 models stay below 0.04 at q8_0. Its claim: Gemma
  degrades uniformly across categories and is *not* "practically lossless" at q8_0. No 12B row and
  no Gemma tool-calling figure; for Qwen, q4_0 damage concentrated in long documents and tool
  calling. Treat as a warning that Gemma 4 is more KV-sensitive than Qwen, not as a 12B measurement.
- Speed: KV quantization saves memory, not time; f16 is faster when it fits. One secondary source
  reports q4_0 decode about 37 % slower at 110k context.
- Gemma 4 mixes sliding-window and global layers, so KV is small to begin with (see below).

## 3. Fit for Grace
- Windows + llama.cpp + CUDA 13: everything above in GGUF/llama.cpp works today. FP8 needs another
  engine (see `inference-engines.md`).
- VRAM freed (12B; measured Grace baseline 9.9 GB at 16k/slot with `--swa-full`, 13.0 GB at 32k/slot):
  - Weights: Q4_0 to IQ4_XS about -0.7 GB; to Q4_K_M about +0.3 GB; Q5_K_M +1.5; Q6_K +2.8;
    Q8_0 +5.7 GB. With 24 GB there is room for Q8_0 plus the 3.5 GB JevK5 checker and Pocket TTS, so
    the decision is quality vs decode speed, not capacity.
  - KV: Grace's own measurement shows `--swa-full` costs about 2 GB at 16k and about 5 GB at 32k vs
    the default, which is the bigger lever than q8 vs f16. Rough size (my estimate): f16 KV to q8_0
    halves it (about -1 to -2.5 GB at 16-32k/slot), q4_0 quarters it. Only worth it if VRAM is
    actually short; it is not today (24 GB card, about 10-13 GB used).
- Tool calling: quantized weights shift argmax on exact JSON arguments; Grace's known failure modes
  (repeated calls, wrong near-miss tool) are the kind a lower-fidelity quant could worsen or a
  higher-fidelity one could slightly relieve. No public evidence either way for Gemma 4 12B, so
  this must be measured.
- Prompt cache: unaffected by weight quant. KV quantization is compatible with cache reuse, but
  cached states were computed under the same dtype, so no mismatch risk within a server run.

## 4. Expected gain, costs
- Q5_K_M/Q6_K/Q8_0 or Unsloth UD-Q4_K_XL for quality: potentially a visible tool-use gain if the
  QAT Q4_0 loses what Unsloth's KLD suggests; unknown size; decode cost 5-45 %. At 50 tok/s,
  Q6_K would be about 40 tok/s, which is still above speech rate; first-sentence latency is
  dominated by TTFT (53 ms), so the voice-path cost is small.
- Unsloth UD-Q4_K_XL: 6.72 GB, same or higher speed, likely better fidelity. Licence: Gemma terms
  apply to derivative weights as for Google's file; Unsloth GGUFs are standard. Risk: their QAT
  conversion is a third-party artefact (verify SHA-256, as done for JevK5).
- KV q8_0 to f16: tiny VRAM cost, possibly a quality gain on Gemma given the KL study, and a small
  speed gain. Nearly free to test.
- NVFP4: no gain expected on decode; no Gemma 4 12B file; skip.

## 5. How we would test it
All on the same llama-server flags, one variable at a time, seeded:
1. `grace-tools-sim` (fix the stale scenarios first, otherwise 57/162 is noise) plus
   `bench/tool_call_check.py` and `grace-chat-bench`, 3 seeds each, for: Google Q4_0 (control),
   Unsloth UD-Q4_K_XL, Q4_K_M, Q5_K_M, Q6_K, Q8_0.
2. Same for `-ctk/-ctv` f16 vs q8_0 (control) vs q8_0/q4_0 split, on the best weight quant.
3. Also run `llama-perplexity`/KLD against Q8_0 as the reference, on Danish and English text, to see
   the 12B's actual sensitivity, since the Unsloth numbers are vendor-run.
4. Record TTFT p50/p95 and decode with the existing latency trace.
Adopt if a quant gives a tools-sim gain beyond seed noise (say 3+ points of the 162 across 3 seeds)
at TTFT p50 <= 80 ms and decode >= 40 tok/s; adopt KV f16 if it measurably helps and VRAM stays
under about 16 GB.

## 6. Verdict
- Weights: **maybe** (cheap to test): QAT Q4_0 may be the weakest 4-bit file for 12B per
  third-party KLD; try UD-Q4_K_XL and Q6_K/Q8_0, since VRAM is plentiful.
- KV: **maybe**: test f16 KV; q4_0 KV **not for us**.
- FP8/NVFP4: **not for us** (no decode gain at batch 1, no Gemma 4 12B file in llama.cpp).
