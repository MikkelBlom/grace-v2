# Making Grace's model smarter and faster: summary and shortlist (R2)

Desk research of 2026-10-23 (R0 sweep + R1 topics, see BRIEF.md). Nothing here is measured on
Grace yet; every item ends in an experiment with our benches. Topic files hold sources.

## Step 0, before any experiment
The tools sim is the yardstick, and today it is not trustworthy: its scenarios demand
`[find_tools, use_tool]` while Grace now goes straight to `use_tool` (57/162, mostly stale
expectations). Fix the scenarios, add held-out lines (loop traps, no-tool lines, near-miss tool
pairs) and take a clean baseline. Without this no result below can be judged.

## Tier 1: cheap, mostly config or harness, likely gains
| # | What | Why | File |
|---|---|---|---|
| 1 | Tool-loop bundle: validate args before running and return the expected schema; describe failures instead of echoing raw args; hard stop on identical tool+args | Our 8-call loops come from `use_tool`'s free `args`; error format effect shown on small models (arXiv 2608.23651) | scaffolding, constrained-decoding |
| 2 | Prompt-cache audit: pin voice to slot 0 and the catalog router to slot 1, warm both at start, run once at `-lv 4` and grep `forcing full prompt re-processing`; pass `--jinja` explicitly | The 1.5 s spikes of live run 8; the router request may evict the voice prefix | prefill-and-prompt-cache, thinking-budget |
| 3 | KV cache f16 vs q8_0 (we run `-ctk/-ctv q8_0`) | An independent study shows Gemma 4 losing more than expected at q8_0 KV | quantization |
| 4 | Weights: Google QAT Q4_0 vs Unsloth Q4_K_XL | Unsloth's KLD 0.133 vs 0.507, top-1 agreement 89 % vs 74 %, slightly smaller (vendor numbers) | quantization |
| 5 | Perceived latency: a pre-synthesised on-hold line while slow tools run; memoise read-only tool results; prefetch read-only tools on partial transcripts (never state-changing ones) | Biggest felt-latency win for tool turns | caching-and-early-action |
| 6 | Laptop: measure a GPU power cap (140-160 W reported within 1-4 % decode); pin the browser to the Arc so YouTube decode stays off the RTX; find why decode is ~50 tok/s when bandwidth allows about twice that | Cheap, also lowers heat for long runs | laptop-hardware |

## Tier 2: worth a real experiment
| # | What | Evidence | Risk |
|---|---|---|---|
| 7 | Gemma 4 MTP drafter (`mtp-gemma-4-12B-it.gguf`, 254 MB, `--spec-type draft-mtp`) | About 1.8x decode on a 3060 blog; in our build already | Untested with `-np 2`, mmproj and the prompt cache; a fork crashed with 2 slots. Test on a second server first |
| 8 | Model bake-off: Gemma 4 26B-A4B (same family, faster), then Qwen3.6-35B-A3B; Gemma 4 31B as the quality ceiling | EuroEval Danish and Tau2 numbers in model-bakeoff.md | VRAM next to vision; Qwen's thinking default and hybrid-attention cache |
| 9 | A 16-64 token thinking budget on tool turns only | One small-model paper: short thinking helps, long hurts | Thinking is switched at the start of the system turn, so it may break the cache |

## Tier 3: bigger investment, biggest possible quality gain
| # | What | Notes |
|---|---|---|
| 10 | LoRA fine-tune for tool use and persona: programmatic data + self-distillation scored by the sim (mix with original data: CacheRL), Unsloth QLoRA, loaded as a runtime LoRA over the official Q4_0 | Only Unsloth confirmed for Gemma 4. Compute: about 1-2 h for 2k examples, a few hours for 10k (an estimate). No Claude/Gemini output: their terms forbid it |
| 11 | Then RL (GRPO, Fission-GRPO for error recovery) with the sim as reward | Only after 10, and only with a trustworthy sim |

## Not for us (and why)
Switching engines (llama.cpp wins for one user on Windows with vision + tools + cache);
NVFP4 (no llama.cpp Gemma 4 12B file, decode unchanged); a small-LLM cascade router (its own
130-300 ms eats the gain); a semantic reply cache (false hits, STT noise); full-duplex models
(no tools, English only); LLMLingua-style prompt compression (breaks the cache); planner/executor,
ReAct, self-consistency on the voice path (latency); FunctionGemma 270M zero-shot (too weak).

## Corrections to the topic files
- context-engineering.md says memory sits at the end of the system prompt and changes per learned
  fact: no, it was moved first and is frozen per session (`MemoryService::prompt_block`).
- We already run `--reasoning off` and `--swa-full`.
