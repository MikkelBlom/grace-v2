# Inference engines: llama.cpp vs ExLlamaV3, vLLM, SGLang, TensorRT-LLM, MLC

Desk research 2026-10-23. Gaps are stated; several cells below were not confirmable from sources
found and are marked "unverified".

## 1. What it is
Grace runs Gemma 4 12B in `llama-server` (llama.cpp b11539, CUDA). The question is whether another
runtime would cut TTFT or raise decode enough to matter for a one-user, batch-1, prefix-cached
voice assistant on Windows.

## 2. Evidence by engine

| | Windows | Gemma 4 | Notes |
|---|---|---|---|
| **llama.cpp** | native (CUDA 12.4/13.3, Vulkan, HIP, OpenVINO builds) | yes; also Gemma 4 **MTP** ([b9549 "llama : add Gemma4 MTP", PR #23398](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b9549)) | OpenAI API + tools + vision (mmproj) + prompt cache; Grace's current stack |
| **ExLlamaV3** | native Python/PyTorch+CUDA, community-used on Windows (unverified for 5090 Laptop) | tensor-parallel changelog mentions Gemma4 (secondary aggregator, unverified against [repo](https://github.com/turboderp-org/exllamav3)) | EXL3 quant format; no OpenAI server of its own (TabbyAPI wraps it); weak multimodal/tool story |
| **vLLM** | no official support; WSL2 or community wheels ([state of support](https://fazm.ai/t/vllm-windows-support-2026)) | yes, day-0, text+vision, native function calling ([vLLM blog](https://vllm.ai/blog/gemma4)); Google ships compressed-tensors QAT for it | built for throughput/batching; automatic prefix caching; OpenAI API with tool parsers |
| **SGLang** | Linux/Docker; WSL2 only | yes (AMD day-0 notes name SGLang, [AMD](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-gemma-4-on-amd-processors-and-gpus.html)) | RadixAttention prefix cache; strong structured output; same batching focus |
| **TensorRT-LLM** | historically a beta native Windows build ([NVIDIA 2023 post](https://blogs.nvidia.com/blog/tensorrt-llm-windows-stable-diffusion-rtx/)); current 2026 Windows/Gemma 4 status **not found** | NVIDIA claimed day-0 Gemma 4 on RTX (secondary: [mayhemcode](https://www.mayhemcode.com/2026/04/gemma-4-amd-gpu-support-vllm-sglang.html)); not confirmed for TRT-LLM on Windows | engine builds are per-GPU-per-shape; NVFP4/FP8 is its strength |
| **MLC LLM** | Vulkan/CUDA on Windows in principle | **not found** | no 2026 evidence of Gemma 4 or of a function-calling server comparable to llama-server |

Numbers: I found **no** published Gemma 4 12B TTFT/decode comparison across engines at batch 1 on
RTX 40/50 series. General knowledge (not sourced here): at batch 1, decode is memory-bandwidth-bound
and llama.cpp's CUDA kernels are within roughly 10-30 % of the best alternatives; ExLlamaV3 and
TensorRT-LLM can lead by that much on dense models, and the gap grows only with batching, which
Grace does not do. Notebookcheck/Phoronix-style engine bake-offs on laptop Blackwell were not
found.

Speculative decoding / MTP: Google's Gemma 4 MTP drafter shares embeddings with the target and
verifies several tokens per pass with identical output
([Google docs](https://ai.google.dev/gemma/docs/mtp/overview.md.txt)). Reported speedups are
community numbers: about 1.85x for +11 % VRAM (one user), "3.34x" on an RTX 6000 PRO with 70-85 %
acceptance (low-reliability news summary,
[aiweekly](https://aiweekly.co/node/2085)). Google notes gains are strongest on dense models and
limited at batch 1 for the 26B-A4B MoE. Since 12B is dense, it is the good case. llama.cpp supports
it (`-md`, `--draft-max`) per [a June 2026 guide](https://knightli.com/es/2026/06/12/gemma-4-mtp-assistant-llama-cli-speedup/)
(secondary). This is already tracked as issue #26.
NVFP4: llama.cpp has it (see `quantization.md`); vLLM/TRT-LLM have more mature NVFP4, but no Gemma 4
12B NVFP4 checkpoint was found.

## 3. Fit for Grace
- Grace's hard requirements: Windows-native child process, OpenAI tool calling, vision projector,
  prefix cache reuse including Gemma's sliding-window layers (`--swa-full`), cancellation, and
  streaming. llama.cpp meets all of them and is measured: TTFT p50 53 ms, p95 243 ms, 50 tok/s.
- vLLM/SGLang under WSL2: adds a VM boundary (localhost networking, GPU memory shared with the
  Windows desktop; one write-up reports a 5090 collapsing to single-digit tok/s under desktop VRAM
  pressure, [towardsai](https://pub.towardsai.net/my-rtx-5090-fell-from-77-to-7-tokens-per-second-vram-pressure-was-the-culprit-38ecf388772c),
  an anecdote). Grace is also Mikkel's daily machine (YouTube, school) so VRAM contention is real.
  Their wins (continuous batching, paged attention) do not apply to one user, and they reserve most
  VRAM by default (`gpu-memory-utilization`), which fights with side models.
- TRT-LLM: engine build per config, Windows status unclear, vision and swa-aware prefix reuse
  unknown for Gemma 4. High effort, no verified payoff.
- ExLlamaV3: could be faster per token on its EXL3 quants (vendor claims of better quality per bit,
  unverified for Gemma 4), but you lose mmproj vision, the llama-server tool template handling,
  and Danish-tested GGUF flows; wrapper server needed.
- Prefix caching is the thing that matters (live run 8: misses cost up to 1.5 s). llama.cpp's
  caching works but needs `--swa-full` and cache slot affinity; vLLM/SGLang caches are mature but
  also hash-based per block and can evict under VRAM pressure.

## 4. Expected gain, costs
- Switching engines: speculative gain 0-30 % decode, TTFT maybe unchanged (already 53 ms p50, where
  the remaining tail is cache misses, not kernels). Costs: porting the launcher in
  `scripts/grace.ps1`, tool-call parser differences, loss of vision/GGUF parity, WSL2 maintenance.
- Staying and adding MTP in llama.cpp: plausible 1.5-2x decode on a dense 12B (community reports;
  verify), +10 % VRAM, no engine change. This is the high-value lever, not a new engine.
- Real risk with any switch: tool-calling behaviour is chat-template- and sampler-dependent; our
  tools-sim numbers would need re-baselining.

## 5. How we would test it
1. Keep llama-server, run it with the Gemma 4 12B MTP drafter at `--draft-max` 2/3/4/6; measure with
   the latency trace: TTFT p50/p95 (drafting must not add to TTFT), decode tok/s, acceptance rate,
   and tools-sim score (output should be identical in distribution; check at temperature 0 for
   bit-equality of tool calls).
2. Only if MTP fails to deliver, a side-by-side on one alternative: vLLM in WSL2 with the Google
   compressed-tensors QAT checkpoint, same 10 Danish requests x 2 harness as `docs/MODELS.md`,
   on TTFT p50/p95 with a fixed 490-token prefix and decode tok/s, with a browser playing a YouTube
   video to expose desktop VRAM contention.
Adopt another engine only if it beats llama.cpp+MTP by >= 25 % first-piece latency p95 or decode,
without losing vision or tools-sim score.

## 6. Verdict
- Switching engine: **not for us**: batch-1, single-user, Windows-native, vision and cache
  requirements all favour llama.cpp, and no evidence of a decisive gain exists.
- Gemma 4 MTP inside llama.cpp: **adopt-candidate** (belongs to the speculative-decoding topic and
  issue #26; mentioned here because it replaces the reason to switch).
