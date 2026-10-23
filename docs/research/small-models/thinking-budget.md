# Thinking / reasoning budget for Gemma 4

Desk research 2026-10-23. Nothing was run.

## 1. What it is
Gemma 4 has an optional thinking mode. `<|think|>` at the start of the system turn makes it write
reasoning in a `<|channel>thought ... <channel|>` block before answering. Grace runs with
`--reasoning off` (scripts/grace.ps1, MODELS.md), so she never thinks. llama.cpp can toggle it
(`--reasoning`, `--chat-template-kwargs '{"enable_thinking":false}'`), cap it (`--reasoning-budget N`,
`--reasoning-budget-message`) and take the kwargs per request.

## 2. Evidence
- Format rules (Google): thinking is "designed to be enabled at the conversation level" in one system
  turn; drop earlier thoughts from history except inside one turn that makes function calls (there
  they must be kept); 26B/31B may emit a thought channel even when off (an empty thought channel in
  the prompt helps); a "LOW" instruction reportedly cuts thinking tokens about 20%.
  https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
- Unsloth guide: sizes include "12B Unified"; sampling temp 1.0, top_p 0.95, top_k 64; use
  llama-server for the kwarg; its tables do not say whether thinking was on.
  https://unsloth.ai/docs/models/gemma-4.md
- Budget flags (llama.cpp PR #13771): 0 = off, -1 = unlimited. Builds after about 2026-04-10 turned
  the budget sampler on for Gemma 4 / Qwen 3.5 with an unlimited default, so set it explicitly
  (issue report: https://github.com/antoinezambelli/forge/issues/54). The forced-stop message has had
  regressions (https://github.com/ggml-org/llama.cpp/issues/22717), and a server-side budget
  overrides the per-request one in at least one integration
  (https://newreleases.io/project/github/goruck/home-generative-agent/release/v3.35.0).
  All "verify on b11539".
- Gains on tool use, small models:
  - "Brief Is Better": Qwen2.5-1.5B, BFCL v3 multiple, 200 tasks. 32 reasoning tokens took accuracy
    44.0% -> 64.0%; 256 tokens FELL to 25.0%; optimum 8-16 tokens. Mechanism: a short think works as
    function routing (wrong-function errors 30.5% -> 1.5%). A fixed template "Function: [name] /
    Key args: [...]" matched it. One small model, one benchmark.
    https://arxiv.org/abs/2604.02155
  - Gemma 4 12B BFCL / tau2 with vs without thinking: not found. A 69.0% tau2 average for a "12B" on
    an aggregator page could not be verified and has no thinking label; not used.
- Latency: at about 50 tok/s, 20 thinking tokens = 0.4 s, 100 = 2 s, all before the first spoken
  word. Thoughts arrive as `reasoning_content`, which grace-llm already keeps out of speech.

## 3. Fit for Grace
- Prefix cache is the hard part. `<|think|>` is at the start of the system turn, so a thinking
  request has a different prefix: system prompt + tool list re-evaluated (live run 8: misses cost up
  to 1.5 s). Possible: dedicate one of the two slots (`-np 2`) to thinking requests so each prefix
  stays warm (relies on llama-server picking the slot by prefix similarity; unverified).
- A hidden plan without the switch: a one-line plan would be spoken unless stripped by the engine.
  Better is the FR-CoT shape on the tool call itself; `find_tools` suggestions already act as that
  routing step, which may be why suggestions lifted the sim from 18/74 to 57/162.
- "Think only before multi-step tool tasks": the router (JevK5 4B, 392 ms p50) could gate it, but the
  gate costs about as much as the budget it saves; simpler gate: only when suggestions fired.
- In the tool loop the engine must pass `reasoning_content` back with the assistant tool-call
  message (tools.rs pushes empty content), otherwise the thought is lost between calls.

### Config audit: what b11539 does, and what to set (from R0)
- Reported (forge issue above, a single user's report): builds after about 2026-04-10 register Gemma 4's
  thinking tags and create a reasoning-budget sampler with an UNLIMITED default; reports of silent hangs,
  and of tool calls landing inside `reasoning_content` instead of `tool_calls`.
- b11539 is far past that date (a b9982 release note already discusses per-request budget fixes, so
  b11539 includes them), so assume the behaviour is present. I could not find b11539's notes or its
  `--help` text online. Unverified: what the default is for Gemma 4 with `--reasoning off` set.
- Our command line already has `--reasoning off` and scripts/grace.ps1 does not pass `--jinja` in the args I read (newer builds may
  default it on; I could not confirm for b11539). Check that the
  running server logs `Chat format:` for the Gemma 4 parser (not `Generic`) and that `/props` shows a
  tool-capable template. Verify, don't assume.
- Recommendation: set explicitly `--jinja --reasoning off --reasoning-budget 0` (belt and braces;
  confirm each against `llama-server --help` on b11539, since `--reasoning` and `--reasoning-budget`
  interplay is what changed). Then run one check: 162-scenario sim, grep traces for any non-empty
  `reasoning_content` and any tool call text inside it. If tool calls ever sit in reasoning, the
  engine should read them as calls (a tolerant fallback), not drop them.

## 4. Expected gain and risks
- Gain: uncertain. The paper's +20 points is for a 1.5B model starting at 44%; a 12B starts higher,
  so a guess of +3 to +10 points on tool scenarios, mostly near-miss picks and argument values.
- Cost: +0.4 to 2 s before first audio on tool turns, cache-miss risk, thought leakage into TTS
  (Gemma 4 can emit thoughts when off), longer replies drifting into the 800 ms budget. Long
  thinking can make results worse.

## 5. How we would test
Tools sim, 162 scenarios x 3 runs, arms: off (baseline), budget 32, budget 64, thinking enabled
per request only on turns where suggestions fired. Record pass rate, 8-loops, time to first tool
call and first audio, cache hit (`cache_n` vs `prompt_n`), thought leaks into speech. Adopt if pass
rate +5 points or loops halved at <= +500 ms p50 on tool turns, with no leaks and a working cut-off.

## 6. Verdict
maybe: a 16-64 token budget on tool turns only is worth one bench arm (short helps, long hurts);
general thinking is not for us (latency target), and the cache split must be solved first.
