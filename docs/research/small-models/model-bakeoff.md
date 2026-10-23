# Model bake-off candidates: what could replace or join Gemma 4 12B

Desk research, 2026-10-23. Nothing was downloaded or run. Anything marked **vendor** comes from the
model's own card; **EuroEval** is read from the public leaderboard data; **est.** is my own
arithmetic, not a measurement. Compare with `docs/MODELS.md` (Gemma 4 12B: Tau2 69.0 vendor,
about 50 tok/s, 13 GB at `-c 32768`).

## 1. What it is

A sweep of open-weight models from 3B to about 32B that were released or updated in 2025-2026,
checked against what Grace needs from her main model: reliable native tool calls over ~130 tools,
short English replies that follow style rules, and good *understanding* of Danish transcripts
(replies are English, so Danish generation matters less than Danish reading). Latency is the
product, so decode speed and prompt-cache behaviour count as much as quality.

## 2. Evidence

### How to read the Danish column
EuroEval Danish "rank" is a mean rank, lower is better, from
[danish_all_simplified.csv](https://raw.githubusercontent.com/EuroEval/leaderboards/main/leaderboards/danish_all_simplified.csv)
(snapshot I could read, which includes Gemma 4 and Qwen3.5 but **not Qwen3.6, Qwen3.8, Granite 4.2,
Muse Glimmer; and Gemma 4 12B did not appear**). Context for scale: Gemini 3 Pro 1.15,
Claude Sonnet 4.5 1.20-1.24, GPT-5.4-mini 1.26-1.47, Gemma 3 12B 1.49. Gemma 4 31B is #2 overall and
#1 open-weight, also confirmed by [Berget](https://berget.ai/blog/gemma-4-general-availability)
(a hosting vendor relaying EuroEval). Reasoning-type models are scored with whatever mode
EuroEval used, which may not be our thinking-off setting.

### Candidate table

Q4 VRAM is est.: weights at Q4 plus KV for 32k context in total (2 slots); hybrid-attention Qwen
models have small KV, Gemma needs `--swa-full` (+~3 GB for 12B). All "decode" classes are est.
relative to Gemma 12B's measured 49 tok/s on this laptop; the only measured outside figure I found
is ~271 tok/s for Qwen3.6-35B-A3B Q4 on a desktop RTX 5090
([community dataset](https://huggingface.co/datasets/omegaprime669/rtx-5090-benchmarks)), and a
3090 test where Gemma 4 26B-A4B was ~3x faster than dense Qwen3.6-27B (see Sources).

| Model (date) | Size / active | Licence | Vision | Q4 VRAM @32k | Tool scores (vendor unless noted) | IFEval / instruction | Danish (EuroEval rank) | Decode class |
|---|---|---|---|---|---|---|---|---|
| **Gemma 4 12B** (Jun 2026, **current**) | 12B dense | Apache 2.0 | image + audio | ~13 GB measured | Tau2 69.0 | IFEval 97.2 (third-party table, unverified) | not in snapshot; Gemma 3 12B 1.49 | 1x (50 tok/s) |
| **Gemma 4 26B-A4B** (Apr 2026) | 25.2B / 3.8B | Apache 2.0 | image | ~17-19 | Tau2 68.2 | none published | **1.40** | fast, 2-3x est. |
| **Gemma 4 31B** (Apr 2026) | 30.7B dense | Apache 2.0 | image | ~21-23, tight | Tau2 76.9; 90/90 single and 30/30 multi-step calls in one third-party test | none published | **1.16 (#2 overall)** | slow, ~0.5x est. |
| Gemma 4 E4B (Apr 2026) | 4.5B eff. (8B w/ emb.) | Apache 2.0 | image + audio | ~5-6 | Tau2 42 (per MODELS.md) | - | 1.83 | very fast |
| **Qwen3.6-35B-A3B** (Apr 2026) | 35B / 3B | Apache 2.0 | image + video | ~21-23, tight (UD-Q4_K_M weights 20.6 GB) | TAU3 67.2, MCP-Atlas 62.8 (no BFCL, no Tau2). Qwen3.5 sibling: BFCL-V4 67.3, Tau2 81.2 | sibling 3.5: 91.9 | sibling Qwen3.5-35B-A3B **1.35** | very fast |
| **Qwen3.6-27B** (Apr 2026) | 27B dense | Apache 2.0 | image + video | ~19-21 | no tool numbers on card; Qwen3.5-27B: BFCL-V4 68.5, Tau2 79.0 | sibling 3.5: 95.0 | not in snapshot | slow, ~0.5x est. |
| **Qwen3.8-27B** (14 Aug 2026) | 27B dense | Apache 2.0 | image + video | ~19-21 | none on card; IFBench 79.5 | IFBench 79.5 | not in snapshot | slow, thinking on by default |
| **Qwen3.5-9B** (Feb-Mar 2026) | 9B dense | Apache 2.0 | image | ~7-9 | **BFCL-V4 66.1, Tau2 79.1**, MultiChallenge 54.5 | **91.5**, IFBench 64.5 | 1.75 (reasoning mode) | ~1.3x est. |
| Qwen3.5-4B | 4B | Apache 2.0 | image | ~4-5 | BFCL-V4 50.3, Tau2 79.9 | 89.8 | 2.14 | very fast |
| **GLM-4.7-Flash** (Jan 2026) | 30B / ~3B | **MIT** | none | ~19-21 | Tau2 79.5 (vendor; Qwen3-30B-A3B 49.0, gpt-oss-20b 47.7) | - | 1.76 | very fast |
| NVIDIA Nemotron 3 Nano 30B-A3B (Dec 2025) | 30B / 3.5B, Mamba-Transformer MoE | NVIDIA Open Model | none | ~19-21 | BFCL v4 53.8, Tau2 49.0 | IFBench 71.5 | 1.76 (1.81 FP8) | very fast |
| IBM Granite 4.2 30B (Aug 2026) | 30B dense | Apache 2.0 | none | ~19-20 | BFCL v4 61.4, Tau3 62.0 | IFBench 77.2 | Danish not a listed language; not in snapshot | slow |
| Ministral 3 14B (Dec 2025) | 14B (13.5B + 0.4B vision) | Apache 2.0 | image | ~9-10 | native function calling, no scores on card | Arena-Hard 0.551 | 1.76 | ~0.8x |
| Ministral 3 8B / 3B | 8B / 3B | Apache 2.0 | image | 5-6 / 2-3 | no scores | IFEval 69 (third-party aggregator) | 1.79 / 2.23 | fast |
| gpt-oss-20b (Aug 2025) | 21B / 3.6B MoE | Apache 2.0 | none | ~13 (MXFP4) | Tau-Bench retail 54.8, airline 38.0 (high effort, model card) | IFEval 67 (aggregator) | 1.82 (medium effort) | very fast |
| Meta Muse Glimmer 30B (10 Aug 2026) | 30B dense | Apache 2.0 | image | ~19 | none found | - | not in snapshot | slow |
| Mistral Small 4 (Mar 2026) | 119B / ~6B | Apache 2.0 | image | **too big** | - | - | - | out of range |
| Qwen3.8 Flash-Next (Aug 2026) | 125B / 6B | - | - | **too big** | - | - | - | out of range |
| Phi-4 family (14B reasoning-plus), xLAM-2 8B, Hammer 7B | 7-14B | MIT / **research-only (xLAM)** / varies | none | 5-9 | xLAM-2 and Hammer report BFCL v2/v3 only, 2024-25 | - | Phi-4-reasoning-plus 3.33 (bad) | - |

Not found: any 2026 Phi (no Phi-5 hit), any new Llama 4 small variant (Scout is 109B, out of range),
SmolLM at a useful size for this role (small, English-centric; not researched further), or a
2026 xLAM/Hammer successor.

### llama.cpp, tool calling and chat templates (the part that decides effort)
- **Gemma 4:** already running on b11539 with native function calling, `--reasoning off`,
  `--swa-full` (MODELS.md). The MoE and 31B should use the same template and parser. Lowest risk.
- **Qwen3.5 / 3.6 / 3.8:** GGUFs and CUDA decode work (the 5090 dataset above). Tool calls use an
  XML-style `<function=...>` format; community reports of tool-call breakage on llama.cpp come from
  parser-vs-template mismatches and mid-conversation system messages, fixed by newer builds or
  patched templates ([Moore2877 template notes](https://huggingface.co/Moore2877/Qwen-Agentic-Chat-Template-llamacpp),
  [mlx-lm issue 1293](https://github.com/ml-explore/mlx-lm/issues/1293) shows the same class of bug
  on MLX). Treat as "works on a recent build, verify with our 130-tool prompt". **Thinking is on by
  default (Qwen3.8 even sets `reasoning_effort` xhigh and `preserve_thinking`)**: must be switched
  off for the voice path, and the tools sim must run with it off.
- **Prompt cache risk for the Qwen family:** the hybrid linear-attention layers (also why the NPU
  could not run Qwen3.5) keep recurrent state, and llama.cpp's reuse of a cached prefix works through
  checkpoints for such models, not by trimming. Unverified for b11539, but it is exactly the
  1.5 s cache-miss failure Grace saw on Gemma. **Measure first.**
- **GLM-4.7-Flash:** an early llama.cpp bug (wrong gating function) caused looping and bad tool
  output until GGUFs were rebuilt; re-download and use Unsloth's tool-calling sampling
  ([Unsloth guide](https://unsloth.ai/docs/models/glm-4.7-flash)). Mind that looping is already
  Grace's top failure.
- **Nemotron 3 Nano, Granite 4.2:** run in llama.cpp via community GGUFs; template details not on
  the cards.

### English reply quality and style
No trustworthy head-to-head exists for "short casual English reply that obeys style rules".
Proxies: Arena-Hard-V2 (Nemotron 67.7, Granite 4.2 67.9, no Gemma/Qwen figures read), IFBench
(Qwen3.5-9B 64.5, Qwen3.8-27B 79.5, Granite 4.2 77.2, Nemotron 71.5). Arena Elo ~1452 for Gemma 4
31B (3rd) is a style-preference score and can be gamed. Qwen models are known for long, markdown-heavy
answers, which our chat bench's length and "speakable" checks will catch. Gemma 12B already scores
0.95-0.97 there, so the bar is high.

### Takeaways from the numbers
1. Google's own Tau2 gives **no tool-use reason** to move inside the Gemma family except to the 31B
   (69.0 / 68.2 / 76.9). The 26B-A4B's case is **speed at equal quality**.
2. Qwen's cards claim the best small tool use (9B: BFCL-V4 66.1, Tau2 79.1; 3.5-35B-A3B: 81.2), but
   Qwen3.6 reports TAU3/MCP-Atlas instead, so the 3.6 numbers do not line up with 3.5's. Different
   benchmark versions: don't subtract across them.
3. **Danish:** Gemma 4 31B (1.16) beats everything open; Qwen3.5-35B-A3B (1.35) and Gemma 4 26B-A4B
   (1.40) are next; Qwen3.5-9B (1.75), Ministral, Nemotron, GLM, gpt-oss and E4B cluster at 1.75-1.83.
   Gemma 3 12B (1.49) sits above Qwen3.5-9B, so the 9B likely loses Danish understanding versus our
   current model. Qwen3.6/3.8 Danish is unknown.
4. Tau2/BFCL are English, API-style tests; Grace's failures (repeat-call loops, no-tool answers,
   near-miss tool) are about a 130-tool catalogue and Danish phrasing. Only our sim settles it.

## 3. Fit for Grace

All Apache/MIT candidates are fine for local use. Excluded on fit: Mistral Small 4 and Qwen3.8
Flash-Next (too big for 24 GB), xLAM-2 (research-only licence, old), Phi-4 (weak Danish), Granite 4.2
(Danish not supported, no vision, dense 30B slow), Nemotron 3 Nano (lower tool scores than peers, custom
licence, no vision). Vision matters because Grace has an mmproj today: Gemma and Qwen keep it,
GLM/Nemotron/Granite/gpt-oss lose it.

VRAM plan: Pocket TTS and edda are not on the 24 GB budget in the way the LLM is (TTS on CPU; edda int8
CT2 ~2 GB, est.). A 22-23 GB LLM (Qwen3.6-35B-A3B, Gemma 31B) leaves almost nothing for Windows,
the compositor, edda, and a second slot; Q4 with a smaller `-np 1`/16k context or lower-bit quants
would be needed. 17-20 GB candidates fit.

## 4. Expected gain, costs and risks

- **Speed:** MoE with ~3-4B active (Gemma 26B-A4B, Qwen3.6-35B-A3B, GLM-4.7-Flash) should decode
  2-4x faster than the 12B dense (est.; real figures from the 5090 dataset suggest the Qwen MoE is
  >100 tok/s). First sentence arrives sooner on tool-result summaries and long replies. TTFT is
  prefill-bound and probably does not change.
- **Quality:** possible tool-use gain with Qwen 3.5/3.6 family and Gemma 31B; possible *loss* of Danish
  understanding on the Qwen 9B and weaker non-Gemma models.
- **Costs:** a 2x larger weight file loads slower and needs a bigger warm-up; the 31B and 35B-A3B
  squeeze VRAM; Qwen family adds chat-template and thinking-mode work and the cache question; every
  swap invalidates the tuned prompts (system.md was tuned on Gemma's habits: chat bench 0.95-0.97,
  style bench 18/18).

## 5. The bake-off

**Set-up (all candidates identical).** Same llama.cpp build as production (re-test on a newer one
only for failures that look like parser bugs), `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -np 2`
(16k per slot; fall back to `-np 1` if VRAM is over ~22 GB), thinking/reasoning **off**, vendor
sampling for non-thinking where given, `--jinja` with the stock template first, patched template only
as a labelled second run. Quant: Google's QAT Q4_0 for Gemma; Unsloth UD-Q4_K_M (or Q4_K_M) for
the rest; record file SHA-256. Fixed system prompt and tool list so the prefix is cacheable. The
**judge stays Gemma 4 12B at temperature 0** for every candidate (so no model grades itself and
runs stay comparable with MODELS.md); also report hard checks alone.

**Shortlist** (order = priority):

1. **Gemma 4 26B-A4B**: same family, same template, Danish 1.40, 2-3x decode. Tests "equal quality, faster".
2. **Qwen3.6-35B-A3B**: fastest Danish-credible candidate with the best agent scores in range; the
   likely tool-use winner and the biggest risk (VRAM, cache, template).
3. **Gemma 4 31B**: the quality ceiling for Danish and Tau2 76.9; tells us how much a slower model
   buys. Run at `-np 1` if needed.
4. **Qwen3.5-9B**: BFCL-V4 66.1 / Tau2 79.1 at 7-9 GB, vision, leaves 15 GB free. Danish 1.75 is the
   question. Doubles as main candidate and as the harness-shape test for the Qwen family.
5. **GLM-4.7-Flash** (optional, wildcard): Tau2 79.5 vendor, MIT, 3B active, but no vision and a
   fresh llama.cpp bug history; run only if 1-2 do not pan out.
6. **Qwen3.6-27B or Qwen3.8-27B** (one of them, ceiling run): newest dense Qwen; skip if 35B-A3B
   already wins. Qwen3.8 is the newest but has no tool numbers or Danish data and thinking defaults on.

Not shortlisted: Nemotron 3 Nano, Granite 4.2, gpt-oss-20b, Ministral 3 14B (all ≥1.76 Danish and
lower or unpublished tool scores), Muse Glimmer (no tool data, dense 30B), E4B/E2B as main.

**Stages** (stop a candidate as soon as it fails a gate):

- **A. Load and cache sanity.** Loads at the target context, VRAM at idle and during decode, server
  returns tool calls in OpenAI form for the 130-tool prefix. Then 20 requests with the same prefix:
  TTFT p50/p95 and tokens re-prefilled per request (MODELS.md baseline: TTFT 53 ms p50, 243 p95; no
  re-read over 1,500 tokens except at scenario start). Qwen candidates: also run the sequence
  user → tool call → tool result → reply and count prefix re-reads.
- **B. Tools sim.** `grace-tools-sim` on the current scenarios (34 scenarios / 44 lines × 3 seeds,
  Gemma baseline 130/132 with english 132/132), after the stale scenarios are refreshed, plus the
  failure cases from the brief: a loop-trap set (tool results that tempt repeated calls),
  no-tool-needed lines, and near-miss tool pairs. Also `decide_bench` tool-choice (Gemma 27/30 Danish,
  29/30 English) read through native calling instead of letter log-probs, with the one-line catalog
  selection step run by the candidate.
- **C. Chat bench.** `grace-chat-bench --tools`, 16 scenarios × 3 seeds, Gemma baseline 0.938-0.966
  (hard 1107/1122). Report hard checks, judge, English-reply rate (scenario 14, 15/15), "quiet
  when told" 21/21, length caps, `<|channel>`-style token leaks.
- **D. Style bench.** `grace-style-bench` 8 scenarios × 3 seeds, learning on; baseline 18/18
  learned, 2/144 false changes, 265/270 checks. Classifier and replier both on the candidate.
- **E. Latency.** From traces on the live engine path with Pocket TTS running: TTFT, first
  speakable piece (baseline 239 ms p50, 419 p95), decode tok/s alone and while TTS and edda run
  (baseline 49 tok/s), total time for a 3-tool turn, peak VRAM, and p95 of a turn after a cache
  miss. Memory bench (`grace-memory-bench`, 217-225/246) as a regression check only for the
  finalists, since extraction prompts are model-sensitive.

**Scores that would justify a switch** (relative to Gemma 12B measured the same day, noise: chat
±0.01, tools ±1-2 checks, memory ±1-4 per category):

- *Faster, same quality (26B-A4B, 35B-A3B, GLM):* tools sim ≥ 129/132 and english 132/132; chat
  ≥ 0.94 with hard ≥ 99.0%; style 18/18 with ≤ 3/144 false changes; decode ≥ 1.8x (≥ 90 tok/s)
  while TTS runs; TTFT p50 ≤ 100 ms with no extra re-prefill; Danish tool choice ≥ 27/30.
- *Smarter (31B, Qwen ceiling runs):* tools sim fewer loop/no-tool/near-miss failures on the new
  hard set by ≥ 25% with no regression elsewhere, chat ≥ 0.96, and decode ≥ 30 tok/s; the first
  sentence still leaves under 800 ms end to end.
- *Reject:* any Danish tool-choice drop of more than 2/30, any English-reply failure, a cache re-read
  per turn, VRAM over 22 GB at the live context, or an unsolved template issue after trying the stock
  and one patched template.
- *Sensible default:* if nothing beats 12B on quality and none is 1.8x faster, keep Gemma 4 12B.

## 6. Side-model notes (router, checker, summariser)

- **Router/checker** (yes/no decisions, now JevK5 4B INT8 on the Arc iGPU, 148/154): **Qwen3.5-4B**
  has the same base family (BFCL-V4 50.3, Tau2 79.9, Danish 2.14, weaker than E4B 1.83 on Danish);
  **Gemma 4 E4B** would run through the same llama-server (Tau2 42, Danish 1.83); **Ministral 3 3B/8B**
  are cheap but with no tool scores. Judged against `decide_bench` 154 items, none is likely to beat
  JevK5; the useful experiment is E4B as an on-GPU fallback when the Arc is busy.
- **Summariser / memory extractor:** a 9B-class model (Qwen3.5-9B) would take the long tool-result
  summarisation and memory extraction off the main model at 7-9 GB, but extraction quality is
  sensitive to prompts (one extra sentence cost 14 memory checks) so test on `grace-memory-bench`.
- **Draft model for speculative decoding:** a small same-vocabulary model of the same family
  (E2B/E4B for Gemma 4, Qwen3.5-0.8B/2B for Qwen) could speed the 31B or 27B runs; mention only as an
  option for the slow ceiling models. Not researched further here (that is issue #26).

## 7. Verdict

**Maybe, with Gemma 4 26B-A4B and Qwen3.6-35B-A3B as adopt-candidates for speed, Gemma 4 31B as
the quality ceiling.** There is no vendor evidence that any model in range is much better at tool
calls than Gemma 4 12B except Qwen's own cards (different benchmarks, English only) and GLM's
Tau2 claim; the case for switching is mostly decode speed from small-active MoEs, which our own
traces must confirm. Mikkel's decision: whether to spend bake-off effort on the Qwen family given
its template, thinking and cache risks, or to stay inside Gemma 4 first.

## Sources

- Gemma 4 cards: [12B](https://huggingface.co/google/gemma-4-12B-it), [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it)
- Qwen: [3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); 3.5-27B/35B-A3B BFCL/Tau2/IFEval via a [secondary page quoting the model card](https://dell.huggingface.co/models/Qwen/Qwen3.5-27B) (not read directly, treat as unconfirmed)
- [Nemotron 3 Nano card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [Granite 4.2 30B card](https://huggingface.co/ibm-granite/granite-4.2-30b), [Ministral 3 14B card](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512)
- GLM-4.7-Flash: [Unsloth guide](https://unsloth.ai/docs/models/glm-4.7-flash)
- gpt-oss: [model card](https://arxiv.org/pdf/2508.10925); IFEval via [BenchLM](https://benchlm.ai/compare/gpt-oss-20b-vs-ministral-3-8b) (aggregator)
- Meta Muse Glimmer: [Artificial Analysis](https://artificialanalysis.ai/articles/muse-glimmer)
- Danish: [EuroEval Danish CSV](https://raw.githubusercontent.com/EuroEval/leaderboards/main/leaderboards/danish_all_simplified.csv), [Berget](https://berget.ai/blog/gemma-4-general-availability)
- Overview tables: [BetterClaw Gemma 4 vs Qwen](https://www.betterclaw.io/blog/gemma-4-vs-qwen-3-5), [Onyx leaderboard](https://onyx.app/self-hosted-llm-leaderboard) (IFEval 97.2 for Gemma 12B comes from here; unverified)
- Speed: [RTX 5090 community dataset](https://huggingface.co/datasets/omegaprime669/rtx-5090-benchmarks) and a 3090 comparison reported by a search snippet only (not opened)
- xLAM-2 / Hammer: [Llama-xLAM-2-8b-fc-r](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)
