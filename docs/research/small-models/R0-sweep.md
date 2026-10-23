# R0 sweep: candidates for a smarter / faster Gemma 4 12B in Grace

Desk research, 2026-10-23. Sources are search snippets plus a few page fetches; many numbers are
vendor claims, forum posts or secondary write-ups, flagged as such. "Verify" means a primary page
was not reached. Topic names refer to the brief's list (speculative, quant, engines, prefill/cache,
cascades, power/NPU, finetune, constrained decoding, thinking, scaffolding, context, bake-off).

## Top NEW candidates (most promising first)

1. **Gemma 4 12B MTP drafter in llama.cpp** (topic: speculative, but Gemma-4-specific, so treat as new).
   Google ships `google/gemma-4-12B-it-assistant` (~0.4B, Apache 2.0), a multi-token-prediction
   drafter that shares the target's KV cache. llama.cpp PR #23398 (merged 2026-06-07) added
   `--spec-type draft-mtp --spec-draft-n-max 4 -md <drafter.gguf>`. Reported 1.5-2x+ on dense Gemma 4
   (31B), none on the MoE. Applies: yes, top item. Caveats: the PR was benchmarked at `-np 1` (Grace
   uses `-np 2`) and with no vision/prompt-cache discussion; quantised KV needed a fix; our build
   b11539 is newer than the merge so the flag should exist. Verify the 12B GGUF drafter (Unsloth
   lists `gemma-4-12B-it-MTP-Q8_0.gguf`) and the draft acceptance on tool-call JSON.
   Sources: https://ai.google.dev/gemma/docs/mtp/overview ,
   https://huggingface.co/google/gemma-4-12B-it-assistant ,
   https://github.com/ggml-org/llama.cpp/pull/23398 ,
   https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/blob/main/MTP/README.md

2. **Speak-first / filler plus speculative (eager) read-only tool calls** (new).
   Have the model emit a short spoken acknowledgement before the tool call and stream it to TTS
   while the tool runs; optionally start read-only tools on a high-confidence router guess before
   the main LLM finishes. Stream's write-up claims 1.5-2 s of perceived responsiveness; FutureAGI
   claims 200-400 ms saved by prefetch (both vendor claims). Applies: yes, directly attacks the
   perceived latency of tool turns; never speculate on state-changing tools (fits the "model can't
   approve itself" rule). https://getstream.io/blog/speculative-tool-calling-voice/ ,
   https://futureagi.com/blog/how-to-optimize-voice-agent-latency-2026/

3. **Tool-error feedback format and loop breaking** (topic: scaffolding, new angle).
   A reported August 2026 study ("Feedback That Backfires", seen only via a secondary blog; verify
   the arXiv original) says putting a raw failed call back into context makes small models repeat it;
   a "do not repeat" line did not help but paraphrasing the failure did. Related: repeated identical
   calls are a documented failure (arXiv 2605.08477), and Fission-GRPO (ACL 2026) trains on error
   recovery. Applies: yes, matches our loop weak spot; the fix is a harness change (rewrite errors,
   hard-stop on identical tool+args, argument pre-validation), no training.
   https://www.beri.net/article/small-model-agent-repeats-failed-tool-call-harness-error-format ,
   https://arxiv.org/pdf/2605.08477 , https://preview.aclanthology.org/ingest-acl/2026.acl-long.1880/

4. **ToolSpec / OoO-Spec: schema-aware speculative decoding for tool calls** (topic: speculative, new twist).
   ToolSpec (arXiv 2604.13519, "up to 4.2x", training-free) drafts the fixed schema tokens with a
   finite-state machine and retrieves past similar calls as drafts; OoO-Spec (arXiv 2608.00814) is a
   follow-up. Applies: maybe; not in llama.cpp, so only the idea transfers. Note llama.cpp's
   `ngram-mod` / ngram-cache speculation gives a poor-man's version for repeated JSON/tool names
   (no measured tool-call numbers found). https://arxiv.org/pdf/2604.13519 ,
   https://arxiv.org/pdf/2608.00814 , https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md

5. **Embedding-based tool retrieval to shrink the catalog / prompt** (topic: context engineering, new).
   Retrieve top 3-5 tools per turn by embedding instead of showing Gemma a ~130-tool catalog;
   arXiv 2603.20313 reports 99.6% fewer tool tokens and 97.1% hit@3 (self-built, 140 queries, 121
   tools); RAG-MCP (2505.03275) reports 73% fewer prompt tokens. Applies: yes, we already have
   harrier-270m on CPU; it could replace or pre-filter the second catalog-pick request. Caveat: a
   variable tool list breaks the fixed-prefix cache unless retrieved tools go after the prefix.
   https://arxiv.org/pdf/2603.20313 , https://www.alphaxiv.org/overview/2505.03275v1 ,
   https://next.redhat.com/2025/12/05/a-practical-approach-to-smart-tool-retrieval-for-enterprise-ai-agents/

6. **Fix prefix-cache misses on SWA models (ctx checkpoints / cache-ram)** (topic: prefill/cache, new detail).
   Gemma 4 uses sliding-window attention, so llama.cpp can only reuse a prefix if a checkpoint exists;
   otherwise it silently reprocesses the whole prompt (logged at trace level). Current flags:
   `--ctx-checkpoints` (default 32), `--cache-ram` (default 8192 MiB). Reports of checkpoint RAM
   growth with Gemma 4. This is a plausible root cause of the 1.5 s misses in live run 8. Applies:
   yes, check server logs at high verbosity and stable-prefix discipline.
   https://github.com/ggml-org/llama.cpp/issues/24587 ,
   https://github.com/ggml-org/llama.cpp/issues/21690 ,
   https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache

7. **Reasoning-budget / thinking defaults in llama.cpp** (topic: thinking, new pitfall).
   Builds after about 2026-04-10 register Gemma 4 thinking tags and create a reasoning-budget
   sampler with an unlimited default; reports of hangs and tool calls landing in `reasoning_content`.
   Run with `--jinja` and an explicit `--reasoning-budget 0` (or `enable_thinking:false`), and check
   what b11539 does by default. Applies: yes, a cheap config audit; also a possible bounded-thinking
   mode for hard tool turns. https://github.com/antoinezambelli/forge/issues/54 ,
   https://github.com/ggml-org/llama.cpp/issues/22717

8. **Rejection-sampling self-distillation with LoRA on our own bench traces** (topic: fine-tune; new recipe details).
   Sample N trajectories per sim scenario from Gemma itself, keep the ones the sim scores as correct,
   SFT a LoRA. Evidence: Step Rejection FT (arXiv 2605.10674) beats plain RFT; DART-SD (2608.18524)
   self-distils multi-turn tool agents; CacheRL (2606.14179) warns filtered data alone cut the
   tool-call rate 83% to 52% unless mixed with original data. Fully inside the licence rules
   (self-generated, programmatic scoring). Applies: yes if tools-sim is fixed first; needs LoRA
   support for Gemma 4 12B in Unsloth and a GGUF LoRA path in llama.cpp. https://arxiv.org/pdf/2605.10674 ,
   https://arxiv.org/pdf/2608.18524 , https://arxiv.org/pdf/2606.14179

9. **GRPO / RL for tool use on small models** (topic: fine-tune, new).
   Papers: GRPO for small-model function calling (2509.04518), ToolExpander for weak models
   (2510.07737), Fission-GRPO error recovery (2601.15625). A 2026 paper (2607.12640, 4B web agent)
   found RL added little over SFT on tasks the model has mostly mastered and was learning-rate
   sensitive. Unsloth supports Gemma 4 RL (E2B Sudoku notebook, 9 GB), no tool-call notebook found.
   Applies: later; only after SFT/RFT, and the sim reward must be trustworthy. https://arxiv.org/abs/2509.04518 ,
   https://arxiv.org/html/2510.07737v1 , https://unsloth.ai/docs/jp/moderu/gemma-4/train

10. **Native NVFP4 on Blackwell (sm_120) prefill** (topic: quant, new data point).
    One guide says build b8967 (2026-04-29) enables native NVFP4 MMQ on sm_120; a community 5090
    test of a 27B NVFP4 showed prefill +68% (pp512 3295 to 5547 t/s), decode unchanged. Applies:
    maybe, prefill only, and we have Google's QAT Q4_0 (quality-calibrated), so an NVFP4 12B would
    have to be as good as QAT; check whether such a GGUF exists. Mobile 5090 is bandwidth-bound for decode.
    https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/

## Full candidate list

### Speculative decoding and drafting
- **EAGLE-3 in llama.cpp** (existing: speculative). Merged around June 2026 (release b9606 mentions
  "spec: add EAGLE3", PR #18039, plus Gemma4 EAGLE3 from RedHatAI). Probably superseded by the
  native MTP drafter for Gemma 4; compare only if MTP acceptance is poor. Verify the supported list.
  https://newreleases.io/project/github/ggml-org/llama.cpp/release/b9606
- **n-gram / ngram-mod self-speculation** (existing). Hash-table n-grams shared across slots;
  flag names changed between versions, check `--help`. No measured tool-call JSON numbers found.
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md
- **Speculative tool calls** (see top 2 and 4; agentic version arXiv 2512.15834, verify).
  https://arxiv.org/pdf/2512.15834

### Prefill, caching, context
- **KV slot save/restore to disk** (existing: prefill/cache). `--slot-save-path` plus
  `POST /slots/{id}?action=save|restore`; a proxy project (hearth-llm) reports 99 s cold to 1.8 s
  restored on a 30k prompt (early build, one project's claim). Applies: modest; useful for restoring
  the fixed system+tool prefix after a server restart, not per turn. Interplay with SWA checkpoints
  unverified. https://github.com/ggml-org/llama.cpp/discussions/18244 , https://pypi.org/project/hearth-llm/
- **Prompt cache only keeps restorable states** (b11642 note; our b11539 is just before it). Decision/
  embedding/rerank requests no longer fill the cache, which may help our second catalog-pick request
  thrash less. Applies: yes, check when upgrading. https://newreleases.io/project/github/ggml-org/llama.cpp/release/b11642
- **Semantic / intent cache for repeated commands** (new). Normalise the transcript, embedding lookup,
  skip the LLM on a hit; vendor claim 400-800 ms saved per hit; documented failure is cross-user
  leakage at a loose threshold. Applies: maybe for idempotent chit-chat/read-only commands only; STT
  noise in Danish makes thresholds hard. https://futureagi.com/blog/what-is-semantic-caching-llms-2026/
- **Tool-call memoisation** (new, no good source found). Cache results of read-only tools (weather,
  calendar) per short TTL. Applies: yes but is a Grace tool-layer feature, not a model technique.
- **Prompt compression (LLMLingua-style)** (new). Not researched deeply; compresses text, which would
  break the fixed prefix and hurt cache hits. Applies: unlikely, prefix caching already makes the
  static part nearly free.
- **NTILC learned tool compression** (arXiv 2606.06566): embeds the registry, model emits a short
  intent, nearest-neighbour picks the tool. Research-grade, claims O(1) registry tokens. Applies: idea
  only (see top 5). https://arxiv.org/pdf/2606.06566

### Fine-tuning and training
- **FunctionGemma 270M fine-tune as a tool router** (new). Google's 270M Gemma 3-based model meant to
  be fine-tuned for function calling; NVIDIA NeMo and Unsloth have recipes; no source tested it for
  choosing among many tools. Applies: maybe as a tiny first-stage router replacing the catalog-pick
  request (batch it, 270M on CPU/iGPU), trained on programmatic data. Verify its licence terms.
  https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/function-calling ,
  https://lmstudio.ai/models/google/functiongemma-270m
- **PRISMS neuron probes / steering** (arXiv 2608.00218, new). Sparse probes detect invalid arguments,
  over-calling and missed calls before generation and can steer. Research-grade, needs model internals
  not exposed by llama-server. Applies: no for now.
  https://www.alphaxiv.org/abs/2608.00218.md
- Step-RFT, DART-SD, CacheRL, GRPO items: see top 8-9.

### Quantization, engines, hardware
- **KV-cache quantization with MTP**: early MTP builds gave 0% acceptance with q8_0 KV until a
  Hadamard-rotation fix; later users still saw lower throughput. Applies: keep f16 KV if MTP is used.
  (existing: quant) https://github.com/ggml-org/llama.cpp/pull/23398
- **Power limit vs decode speed on laptop 5090** (existing: power). One source: 140-160 W decodes within
  1-4% of 175 W, a 95 W mode loses about 28%; boost variance up to +-15% on desktop cards, clock
  locks help benchmarking. Windows `nvidia-smi -lgc` on a laptop is unverified. Applies: make sure the
  OEM "performance" profile and plugged-in power are on during benches; blog-level evidence.
  https://computingforgeeks.com/best-laptop-for-local-llm/ , https://inventivehq.com/blog/gpu-power-limit-tokens-per-watt
- **vLLM / SGLang / TensorRT-LLM** (existing: engines). Not researched in depth; Gemma 4 tool parsers
  in vLLM exist and have had bugs; Windows support is the blocker. Applies: low priority.
  https://github.com/vllm-project/vllm/pull/38847
- **CUDA graphs / flash-attention flags**: no 2026-specific findings; `-fa on` and `-ngl 999` are the
  documented defaults in current Gemma 4 recipes. Applies: confirm `-fa on` in our launch line.

### Constrained decoding and parsing
- **Gemma 4 tool-call format** is its own non-JSON syntax (`<|tool_call>call:name{...}`); reported to
  degrade to plain-text calls under long system prompts, multiple tools and 3-4 sequential calls, and
  to loop on malformed calls across vLLM/llama.cpp/Ollama (single-author claims). Applies: yes,
  explains part of our failure set; a tolerant fallback parser, grammar-constrained retries and a short
  prompt would be the levers. (existing: constrained) https://discuss.huggingface.co/t/gemma-4-bug-fixes-and-research-request/176979 ,
  https://github.com/ggml-org/llama.cpp/issues/22786 , https://docs.vllm.ai/en/v0.19.1/api/vllm/tool_parsers/gemma4_tool_parser/
- No source found on llama.cpp lazy grammars for Gemma 4 specifically; test directly.

### Scaffolding
- **G-STEP finish gate** (arXiv 2605.00136): lightweight check when the model tries to finish decides
  whether to force one more tool step. Applies: maybe, targets "answers without a tool when one fits".
  https://arxiv.org/pdf/2605.00136
- **AgentCheck** (arXiv 2607.11098): fault-injection for MCP tools; retry raised weakest-agent success
  on timeouts from about 30% to 100%, stale data stayed 30-40%. Applies: borrow the fault-injection idea
  for tools-sim. https://www.alphaxiv.org/abs/2607.11098v3
- **Planning horizon in tool calling** (arXiv 2605.08477): questions step-by-step planning for
  data-centric tools. Applies: low. https://arxiv.org/pdf/2605.08477

### Voice-agent architecture
- **Full-duplex speech-to-speech with tool calling**: NVIDIA NemotronLabs VoiceChat 11B (arXiv 2609.21967,
  about 448 ms turn latency, "on-hold" lines during tool calls), PersonaPlex, MoshiRAG (async retrieval),
  DuplexSLA. Applies: not a drop-in (Danish in, local LLM stack); the "on-hold line during a tool call"
  pattern is the transferable part (see top 2). Already noted in FUTURE_IDEAS for PersonaPlex.
  https://arxiv.org/pdf/2609.21967
- **Preemptive generation** (LiveKit `preemptive_generation`, start the LLM on interim STT): vendor-level
  guidance claims big p95 drops. Applies: maybe, needs a cancellable LLM request on partial transcripts and
  a prompt cache that survives cancel. https://futureagi.com/blog/how-to-optimize-livekit-latency-2026/
- **Small model for short turns** (cascade, existing): vendor claim 100-300 ms TTFT saved. Not independent.

### Model bake-off
- Community sentiment (LocalLLaMA snapshots, anecdotal) says Qwen 3.6 handles agentic tool calls better
  than Gemma 4, but part of the Gemma complaints trace to thinking/parser configuration. No hard BFCL
  numbers found for Gemma 4 12B vs Qwen 3.5 9B / Qwen 3.6. Applies: bake-off should run both through
  tools-sim with thinking off and the right parser. https://www.betterclaw.io/blog/gemma-4-12b-vs-qwen-3-5-9b ,
  https://insiderllm.com/guides/function-calling-local-llms/

### Not found / not pursued
Model merging, layer skipping/early exit and steering vectors for serving: no llama.cpp-ready,
Gemma-4-relevant source turned up in this sweep; research-grade at best, treat as not for us. Windows
NPU/iGPU offload of the main model: nothing new beyond the brief's finding.
