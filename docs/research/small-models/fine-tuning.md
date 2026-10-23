# Fine-tuning Gemma 4 12B (LoRA / QLoRA) on the laptop

Desk research, 2026-10-23. Nothing was run. Where a number is my own arithmetic or unverified, it says so.

## 1. What it is
Train a small set of adapter weights (LoRA) on top of a frozen Gemma 4 12B, on examples of the
behaviour we want (right tool, right arguments, Grace's voice, English reply to Danish input).
QLoRA does the same with the frozen base held in 4-bit to save VRAM. The result is an adapter
(~100-400 MB) that is either merged into the base or loaded next to it.

## 2. Evidence

### Which tools support Gemma 4, and VRAM
- Gemma 4 shipped 2026-04-02 under plain Apache 2.0 in five sizes (E2B, E4B, 12B, 26B-A4B, 31B);
  the 12B "Unified" model is encoder-free, so the whole model can be tuned in one pass, and it has
  native function calling ([model card](https://huggingface.co/google/gemma-4-12b-it)).
- **Unsloth** is the only stack I could confirm for Gemma 4 ([guide](https://unsloth.ai/docs/models/gemma-4/train)).
  Its page lists 12B but gives no 12B VRAM figure. Reported numbers on that page: E4B QLoRA 10 GB
  (LoRA 17 GB), 31B QLoRA 22 GB, 26B-A4B LoRA >40 GB (MoE QLoRA not recommended). Interpolating
  by parameter count, 12B QLoRA should be roughly 10-14 GB and bf16 LoRA roughly 28+ GB (does not
  fit). That interpolation is mine, not a measured figure. Unsloth's generic table: 8B QLoRA ~6 GB,
  27B QLoRA ~22 GB ([requirements](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements)).
- Unsloth Studio runs on Windows (PowerShell installer) for NVIDIA GPUs; a WSL install path also exists.
  Unsloth lists known Gemma 4 gotchas (grad-accumulation loss, upstream transformers `num_kv_shared_layers`
  IndexError on 26B/31B, bad logits with `use_cache=False` on E2B/E4B). Same page.
- **Axolotl, TRL/PEFT, torchtune**: my searches found no confirmation of Gemma 4 support for any of
  them. TRL/PEFT works for any model transformers supports, so it likely works once the transformers
  version has Gemma 4 (it does, since Unsloth builds on it), but the 12B-specific multimodal
  wrapper and KV-sharing quirks are unverified. torchtune has been slow to add new families (unverified;
  check its repo). Treat Axolotl/torchtune as "unknown, check before relying".
- Blackwell (sm_120) needs a recent PyTorch/CUDA 12.8+ build; bitsandbytes and flash-attn wheels on
  Windows have historically lagged. Unverified for Gemma 4 specifically; WSL2 with CUDA passthrough is
  the lower-risk path for TRL/Axolotl, Unsloth Studio is the lower-risk path on native Windows.

### Sequence length and throughput
- No published Gemma 4 12B tokens/s on a 5090 Laptop found. I will not invent one. Planning figure
  (my estimate, wide): QLoRA on a ~12B model on a consumer 24 GB Blackwell GPU lands in the
  low thousands of training tokens/s with Unsloth-style kernels; 1-3k tok/s is a sane bracket.
- Sequence length matters because our prompt is a fixed prefix (system prompt + tool list). With loss
  masked to the assistant turn, the prefix still costs compute and activation memory. Train with the
  real prefix (the model must learn under the real prompt) at 4k-8k tokens; Unsloth's gradient
  checkpointing/offload lets 24 GB reach 20-32k on 8B-class QLoRA ([long-context GRPO post](https://unsloth.ai/docs/jp/meru/reinforcement-learning-rl-guide/grpo-long-context)),
  so 8k on 12B is plausible but unmeasured.
- Compute-runtime arithmetic (mine; assumes ~1.5k tokens/example incl. prefix, 2 epochs, 1-3k tok/s):
  - 2k examples: ~6M tokens -> roughly 35 min to 1.7 h
  - 10k examples: ~30M tokens -> roughly 2.8 to 8 h
  - 50k examples: ~150M tokens -> roughly 14 to 40 h
  Shrinking the tool list in training prompts (or caching the shared prefix) cuts this a lot; do not
  do it if inference uses the full list.

### Back to GGUF for llama.cpp
- Community default is merge adapter into the HF base, `convert_hf_to_gguf.py`, then `llama-quantize`
  ([example scripts](https://huggingface.co/evalstate/trl-demo-scripts/blob/main/convert_to_gguf.py)).
  Unsloth exports directly but only offers q4_k_m / q8_0 / f16 ([guide](https://unsloth.ai/docs/models/gemma-4/train)).
- llama.cpp can also load LoRA at runtime (`--lora adapter.gguf`, made with `convert_lora_to_gguf.py`).
  I found no public comparison of runtime-LoRA-over-quantized-base vs merge-then-quantize quality. Runtime LoRA
  keeps Google's exact QAT Q4_0 base untouched and allows swapping/scaling adapters, at some decode cost
  (extra matmuls) and with a caveat that per-slot cache and `-np 2` behaviour should be tested. Unverified here.
- **QAT benefit when merging:** Google's QAT checkpoints are trained so Q4_0 rounding hurts little.
  Merging a LoRA delta into the *unquantized* QAT checkpoint and re-running Q4_0 breaks that alignment: the
  merged weights are no longer the ones QAT optimised the rounding for. Unsloth reports even *naive*
  conversion of the unmodified QAT checkpoint to Q4_0 is lossy: 12B top-1 agreement 74.1% vs 88.8% with their
  method, mean KLD 0.507 vs 0.133 ([Unsloth QAT page](https://unsloth.ai/docs/models/gemma-4/qat)), because
  llama.cpp's Q4_0 uses F16 scales while QAT uses BF16 scales. So a merged re-quantised model will probably
  sit below Google's official Q4_0 on fidelity; the LoRA gain must beat that loss.
- **QAT-aware LoRA** exists in research (e.g. LoRA on a fake-quantised base, QA-LoRA, LR-QAT, EfficientQAT)
  but I found no Gemma 4 / llama.cpp-ready tool. Practical substitute: train QLoRA on the 4-bit base
  (adapter learns to compensate for 4-bit error), then use runtime `--lora` over the official Q4_0 GGUF. Whether
  QLoRA's nf4 base matches Q4_0 error is not guaranteed; we would measure it.

### What tool-calling fine-tuning has achieved
- ToolACE-8B (Llama-3.1-8B, synthetic data from a 26,507-API pool) claimed BFCL state of the art for 8B at ICLR 2025
  ([paper](https://arxiv.org/abs/2409.00920)). Hammer-7b (Qwen2 base, 60k APIGen samples + irrelevance data,
  function masking) and xLAM-2 are similar. A third-party snapshot of BFCL multi-turn lists xLAM-2-8b-fc-r ~70% vs
  ToolACE-2-8B ~38% ([benchmarklist](https://benchmarklist.com/benchmarks/bfcl_v3_multiturn/), not official).
  All are vendor/author claims on BFCL, whose tools are English API schemas, not a 130-tool voice catalog.
- Lesson that transfers: "irrelevance" examples (no tool fits -> answer in text) matter, and so does training
  on multi-turn state. Both are our weak spots (answering without a tool, loops).
- Risks: catastrophic forgetting (LoRA at low rank and low LR forgets less than full fine-tuning, a known
  trade-off in "LoRA Learns Less and Forgets Less", 2024; I did not re-verify its numbers), persona drift
  (tool data in neutral register overwrites Grace's voice unless persona examples are mixed in), and Danish
  understanding loss (if all training inputs are English or clean text, comprehension of Danish STT noise can
  degrade). Mitigations: mix in replay of Danish chat and persona turns, keep rank 16-32, LR ~1e-4 or lower,
  1-2 epochs, evaluate with grace-chat-bench and grace-style-bench before and after. No Gemma-specific forgetting
  numbers found.

### Pausing, checkpoints, shared laptop
- All of Unsloth, TRL and Axolotl save checkpoints every N steps and resume (optimizer state included). Kill and resume is routine.
- The GPU is fully used during a run; llama-server (~13 GB) and training (~10-14 GB QLoRA) cannot both fit in 24 GB with
  comfortable headroom, so Grace must be down while training (unverified margins; it is tight). Training also pins CPU
  and drives laptop thermals; expect fan noise and throttling on a laptop (no data). Practical pattern: run in
  chunks with checkpoints (e.g. overnight, on power), stop llama-server first. Windows desktop compositing uses ~1-2 GB VRAM.

## 3. Fit for Grace
Works on this GPU in principle with Unsloth (native Windows or WSL2). Gemma 4 12B QLoRA is the right scale for 24 GB.
The unresolved part is the return trip: official Q4_0 QAT GGUF -> our own merged GGUF costs fidelity. Prompt-cache
unaffected (tuning changes weights, not the prefix) as long as the training prompt equals the serving prompt.
Danish in / English out: keep Danish STT-style inputs in training, replies English.

## 4. Expected gain and costs
Gain: biggest on known failure modes (loops, no-tool answers, near-miss picks). Reports from small-model tool-use work suggest
double-digit point gains on in-distribution tool benchmarks, but gain on *our* sim is the thing to measure; there's real
overfit risk because the sim is also what we'd train against (see training-data.md). Costs: adapter + merge pipeline to maintain
on each llama.cpp or base-model bump; possible Q4_0 fidelity loss; Grace offline during training; licence is fine (Apache 2.0).

## 5. How we would test it
1. Build train/held-out split of scenarios (see training-data.md). 2. QLoRA 2k examples, rank 16, with Danish and persona replay.
3. Serve two ways: (a) llama-server `--lora` over the official Q4_0, (b) merged + Q4_0/Q4_K_M. 4. Run grace-tools-sim on held-out
scenarios (multiple seeds), grace-chat-bench, grace-style-bench, and latency traces (decode tok/s, TTFT). Adopt if held-out sim
improves clearly beyond seed noise (say >=10 points absolute on the weak categories), chat/style benches drop <= noise, Danish
understanding set does not regress, and decode stays near 50 tok/s.

## 6. Verdict
**maybe, leaning adopt-candidate** for runtime-LoRA on the official Q4_0; the only real unknown is merge/re-quantisation fidelity, which an A/B settles. Tooling other than Unsloth is unconfirmed for Gemma 4.
