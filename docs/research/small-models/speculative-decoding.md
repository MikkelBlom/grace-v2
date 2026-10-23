# Speculative decoding for Gemma 4 12B in llama.cpp

Desk research, 2026-10-23. Nothing was run. Verdict at the bottom.

## 1. What it is
A cheap "drafter" guesses the next few tokens; the 12B checks them all in one forward pass and
keeps the longest correct prefix. Output is identical to normal decoding (at temperature 0; with
sampling it is still distribution-preserving), so quality does not change, only speed. The
drafter can be (a) a separate small model with the same tokenizer (`draft-simple`), (b) a
Multi-Token-Prediction (MTP) "assistant" that reads the target's KV cache and activations
(`draft-mtp`), (c) an EAGLE-3 head (`draft-eagle3`), or (d) no model at all: n-gram lookup in the
prompt and recent output (`ngram-*`).

## 2. Evidence

### Gemma 4 and MTP (the relevant option)
- Google's release log lists Gemma 4 on 2026-03-31 (E2B, E4B, 31B, 26B-A4B), "MTP" drafters on
  2026-04-16 for those four sizes, and "Gemma 4 12B Unified" on 2026-06-03
  ([release log](https://ai.google.dev/gemma/docs/releases?hl=en)). Google's MTP page lists 12B as
  a target and names drafters `<target>-assistant` (e.g. `google/gemma-4-E2B-it-assistant`); the
  drafter is a 4-layer model that reuses the target's KV cache
  ([MTP docs](https://ai.google.dev/gemma/docs/mtp/mtp)). The page gives no speedup numbers and the
  release log does not mention a 12B drafter, so whether an official `gemma-4-12b-it-assistant`
  exists is not confirmed from Google's pages. One user nonetheless ran `mtp-gemma-4-12B-it.gguf`
  (see below).
- Mainline llama.cpp: PR #23398 "Gemma 4 MTP" merged 2026-06-07 for 31B and 26B-A4B; E2B/E4B
  followed ~06-08 ([PR](https://github.com/ggml-org/llama.cpp/pull/23398),
  [E2B/E4B note](https://aiweekly.co/alerts/llamacpp-adds-gemma-4-e2be4b-mtp-assistant-support)).
  Flags: `--spec-type draft-mtp --model-draft <drafter.gguf> --spec-draft-n-max N`
  (`--spec-draft-device` for multi-GPU). The current docs list types `draft-simple`,
  `draft-eagle3`, `draft-dflash`, `draft-dspark`, `draft-mtp`, `ngram-cache`, `ngram-simple`,
  `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`; default `--spec-draft-n-max` 3
  ([spec docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)).
- Only 12B measurement found (one person's blog, RTX 3060 12 GB, build 9574, Unsloth QAT
  Q4_K_XL target, 4-bit drafter, `n-max 4`, q8_0 KV): generation 33.8-39.5 tok/s without, 62-67
  with (about 1.8x) on a coding task; acceptance 75-88% on code, but only 45% average (range
  34-76%) over several days of everyday use, giving roughly 40 tok/s. MTP overhead: ~760 MiB GPU +
  ~325 MiB RAM. Prompt ingestion fell to 70-73% of baseline. That run used `--no-cache-prompt`
  and quality was not evaluated
  ([source](https://lp-soroban.highreso.jp/compute-cluster/blog/mtp-faster-generation.html)).
- 31B/26B numbers (different sizes, still indicative): RTX PRO 6000, Q8 target, 1.74x/1.89x/1.97x
  at n=2/3/4; Q4 target only ~1.5-1.55x; MoE 26B-A4B no gain in the PR author's setup
  ([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)). Acceptance falls with draft
  length: ~89/83/74/64% at n=1/2/3/4 (31B Q8, fork PR numbers, vendor-less single source)
  ([point.free](https://point.free/blog/gemma-4-mtp/)). Strix Halo fork: 31B +86..+266%,
  26B-A4B +22..+45% ([sleepingrobots](http://sleepingrobots.com/dreams/gemma4-mtp-assistant-strix-halo/)).
- Drafter size: ~78M (E2B) to ~470M (31B) params; Q4_K_M drafter files ~310-340 MB for the big
  models. A Q4 drafter was ~13% faster in total than Q8 on an E4B target; avoid <=Q3
  ([point.free](https://point.free/blog/gemma-4-mtp/)).

### Known problems relevant to Grace
- Quantised KV cache: with `-ctk q8_0 -ctv q8_0` one user saw 0% acceptance at merge; fixed later
  by adding a missing Hadamard rotation ([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)).
  Grace runs q8_0 KV, so this must be re-checked on b11539+.
- Parallel slots: a fork report says `--n-parallel 2` crashed in `llm_build_gemma4_mtp` because the
  draft step handles one token column; unconfirmed whether mainline fixed it
  ([search result](https://aiweekly.co/alerts/gemma-4-mtp-support-proposed-for-llamacpp)). A
  Qwen MTP write-up says MTP is silently disabled with `-np > 1` and with `--cache-reuse`
  ([HF space](https://victor-mtp-on-hf-endpoints.static.hf.space/index.html)); not verified for
  Gemma. Every published Gemma benchmark used one slot.
- Also reported: CUDA-graph cache growth/crash under KV saturation (workaround
  `GGML_CUDA_DISABLE_GRAPHS=1`), MTP silently disabled when VRAM is short, multimodal (`--mmproj`)
  untested. A 2025 llama.cpp commit disabled speculative decoding for SWA models
  ([#13970](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/363757628848a27a435bbf22ff9476e9aeda5f40));
  Gemma MTP evidently works now, but Grace's `--swa-full` combination is untested by anyone.

### Classic draft model (small Gemma 4 as drafter)
Gemma 4 E2B/E4B are the same family, but Gemma 4's family-shared tokenizer is not confirmed by any
source I found; check vocab equality before trying. Reported results are poor: on Strix Halo an
E2B draft gave 26B-A4B -27% and 31B +100% ([sleepingrobots](http://sleepingrobots.com/dreams/gemma4-mtp-assistant-strix-halo/)).
E2B costs ~3+ GB VRAM and its own prefill. MTP is the better version of the same idea.

### EAGLE-3 / Medusa
llama.cpp has `draft-eagle3`, and NVIDIA NeMo AutoModel can train an EAGLE-3 head for Gemma 4
targets ([NeMo docs](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/speculative/eagle/draft_gemma.md)),
but I found no published Gemma 4 12B EAGLE-3 head or GGUF. A commenter reported ~1.4x with EAGLE-3
on 26B-A4B. Medusa-style heads: nothing for Gemma 4. Training our own head is out of scope and
Google's MTP drafters replace it.

### n-gram / prompt lookup
`ngram-mod` (shared hash pool across slots, defaults n-match 24, n-min 48, n-max 64) and
`ngram-cache`/`ngram-map-k` need output that repeats the prompt. Measured example (Qwen, long
context synthesis): ngram acceptance 32%, decode -17%
([HF space](https://victor-mtp-on-hf-endpoints.static.hf.space/index.html)). Grace's replies are
short, freshly worded English from a Danish question; little repeats. Possible niche: tool-call
JSON that copies argument strings from the prompt or a previous call, and read-back of fetched text.

### Update (coordinator's R0 findings, checked)
- **A 12B drafter exists as GGUF.** Unsloth ships it inside its 12B repos: `mtp-gemma-4-12B-it.gguf`
  (Q4_0, 254 MB, "smart Q4_0") at the root of
  [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)
  (which is the QAT Q4_0 family Grace uses), and Q8_0 465 MB / BF16 / F16 862 MB in the `MTP/`
  folder of [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/blob/main/MTP/README.md).
  Its README says it works with any quant of the 12B and with quantised KV cache, recommends
  Q8_0, and quotes (vendor, one B200, Q4_K_M target) 52 -> 162 tok/s at 0.70 acceptance.
  Command from the card: `--spec-type draft-mtp --spec-draft-n-max 4 -ngl 999 -fa on`; `-hf` finds the
  drafter automatically, otherwise pass `-md`. (My earlier "unconfirmed" note on the
  `google/gemma-4-12B-it-assistant` drafter is superseded; Google's 0.4B figure I could not
  verify, the Unsloth Q4_0 file at 254 MB is consistent with a few hundred million parameters.)
- **Builds:** the Unsloth card requires a build after 2026-06-07 (PR #23398). b11539 is far
  later (b9574 was a June build in the 12B blog), so the merged code is in. Later fixes are not
  enumerated anywhere I found; read the b11539 release notes / `git log` for `mtp`.
- **Quantised-KV fix:** reported at merge as 0% acceptance with q8_0 KV, fixed by a Hadamard
  rotation of Q ([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)); Unsloth says
  quantised KV is supported. Confirm in Grace's log that acceptance is non-zero.
- **`-np 2`, mmproj, prompt cache: no source documents any of them for Gemma MTP.** Everything
  published is `--parallel 1`; the only parallel report is a fork crash (ggml_reshape_3d in
  `llm_build_gemma4_mtp`) and the 12B blog disabled the prompt cache. Treat as unknown; test
  first with each feature added separately.
- **Acceptance on JSON:** no measurement found. Indirect: code (strict grammar) gave 75-88%
  acceptance versus 34-76% in everyday chat on the same drafter, so tool-call JSON should sit near
  the code end (inference). Short replies cap the benefit: with n-max 3-4 the first draft
  tokens carry most of the gain.
- **`ngram-mod`** (`--spec-type ngram-mod`, or `--spec-default`): no model, shared hash pool across
  slots (so `-np 2` is fine by design), defaults n-match 24 / n-min 48 / n-max 64. Those lengths
  are tuned for long repeated text; 48+ draft tokens are pointless for Grace's one-sentence
  replies and short JSON. It can be combined with MTP (`--spec-type ngram-mod,draft-mtp` style,
  per the docs, "draftless takes precedence"), which would let ngram handle copy-from-prompt
  arguments. Low priority.
- **Schema-aware speculation papers** (idea transfers, code does not):
  [ToolSpec 2604.13519](https://arxiv.org/abs/2604.13519) fills schema tokens deterministically via
  a finite-state machine and drafts variable fields from retrieved past calls and claims up to
  4.2x (paper's claim, training-free per its framing, abstract only read);
  [OoO-Spec 2608.00814](https://arxiv.org/abs/2608.00814) adds a Qwen3-0.6B LoRA sidecar that
  predicts function and arguments in parallel, 2.46-5.34x over autoregressive (paper's claim,
  greedy batch-one, Qwen/Llama targets). Neither is in llama.cpp. The transferable, cheap parts
  for Grace: (a) the fixed JSON scaffold of a tool call (`{"name": ..., "arguments": {`) is already
  highly predictable to MTP/ngram; (b) grammar-constrained output via llama.cpp's `grammar` /
  `json_schema` already removes scaffold-token errors but not decode time; (c) a retrieval of
  previous calls to the same tool as `ngram` seed needs custom server work. Not worth building
  unless tool-call decode time shows up as a real share of turn latency in traces.

## 3. Fit for Grace
- Windows/CUDA/llama.cpp: build b11539 is newer than the build-9574 12B report, so MTP code is
  present. Need the 12B drafter GGUF (convert from `-assistant` weights or find a published GGUF)
  and `--spec-type draft-mtp -md <drafter> --spec-draft-n-max 2..3`.
- `-np 2`: unknown/risky (see above). Fallback: run a second llama-server instance only for the
  main voice reply with `-np 1` (costs a duplicate 13 GB, too much) or give up the router slot.
- Prompt cache: the one 12B report disabled it. Grace needs it (cache misses cost 1.5 s). MTP's
  prefill slowdown (-27..-30%) applies to the cold-prefix case on top of that.
- Tool-call JSON: verification makes output identical, so schema correctness is unaffected; JSON is
  highly predictable, so acceptance should be high there (inference, not measured).
- Vision (`--mmproj`, `-ub 1120`): untested with MTP.
- Danish in / English out: only English tokens are drafted; no concern.

## 4. Expected gain and costs
Decode 50 tok/s -> roughly 65-90 tok/s (1.3-1.8x) if acceptance is 45-80%. This matters for long
replies and tool-call JSON, not for TTFT: the 800 ms target is dominated by STT, prefill and the
first sentence. A shorter reply finishing sooner still helps sentence-by-sentence TTS gaps only if
decode is currently the limiter (check the traces: is TTS ever waiting on the next sentence?).
Costs: ~1.1 GB VRAM+RAM, slower prefill, new failure modes (crashes, silent disable), and a build
we have to babysit. Quality: none lost by design.

## 5. How we would test
1. Obtain the 12B drafter; start a second llama-server on another port with exactly Grace's flags
   plus `--spec-type draft-mtp -md ... --spec-draft-n-max 3`, `-np 2`, q8_0 KV, `--swa-full`,
   mmproj. Pass criterion zero: no crash, server log shows acceptance > 0 (the q8 KV 0% bug).
2. `grace-tools-sim` and `grace-chat-bench` replayed against both servers at temperature 0:
   outputs should be identical-ish (tool-sim score within noise), log decode tok/s, acceptance.
3. Latency traces: TTFT p50/p95 with the prompt cache on (the 341 ms p50 baseline) must not worsen
   by more than ~30 ms; decode tok/s median on tool-call turns and on 2-3 sentence replies.
Adopt if: decode >= 1.3x on real Grace turns, TTFT p50 unchanged, tool-sim unchanged, no crash over
a long soak with both slots used, and `-np 2` works.

## 6. Verdict
**Maybe.** MTP is real, lossless and now in mainline, but the only 12B evidence is one RTX 3060
blog with the prompt cache off and a single slot; Grace's `-np 2` + `--swa-full` + q8 KV + mmproj
combination is exactly where the reported bugs live, and decode speed is probably not the voice
bottleneck. n-gram and EAGLE-3/Medusa and a separate E2B drafter: **not for us**.
