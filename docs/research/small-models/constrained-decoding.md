# Constrained decoding for tool calls (GBNF, JSON schema, lazy grammars)

Desk research 2026-10-23. Nothing was run.

## 1. What it is
llama.cpp can force the model's output to follow a grammar (GBNF). A JSON schema or
`response_format` is compiled into a GBNF grammar and applied at sampling time: tokens that would
break the grammar get probability 0. With `--jinja` the server also renders the tool list through
the model's chat template and parses the model's tool-call text back into OpenAI `tool_calls`.
For some formats it builds a *lazy* grammar: sampling is free until a trigger (the format's
"tool call starts" marker) appears, then the grammar takes over for the call itself.
Grace already uses the plain `response_format` path for decisions (`crates/grace-llm/src/client.rs`,
schema-constrained completion).

## 2. Evidence
- llama.cpp function-calling doc: `--jinja` enables template-based tool calling; native handlers per
  format, otherwise a "Generic" handler that costs more tokens; parallel calls off by default;
  extreme KV quantisation (e.g. `-ctk q4_0`) can hurt tool calling. The doc does not mention
  Gemma 4, lazy grammars or `tool_choice` (it lags the code).
  https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Gemma 4 has a dedicated parser in llama.cpp: PR "common : add gemma 4 specialized parser" (#21418;
  seen only via a mirror/search snippet, diff not read).
  https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/21418/overview
  Gemma 4's native call format is not JSON (`<|tool_call>` ... `<tool_call|>`, strings wrapped in
  `<|"|>`), so the parser converts it to JSON:
  https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
- Lazy-grammar mechanism (trigger patterns) came in llama.cpp #12034 (mirror):
  https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/669912d9a5bf927312c553332ff997f0a99da8fb
  I could NOT confirm whether the Gemma 4 path attaches a grammar at all or only parses after the
  fact. That needs a read of `common/chat.cpp` at b11539.
- Gemma 4 + llama.cpp tool-call trouble on record: a Windows report where a correct call came back
  as plain content instead of `tool_calls` (https://github.com/ggml-org/llama.cpp/issues/22786);
  a forum thread (one person's claim) that under long context + reasoning Gemma 4 malforms calls
  and loops on them: https://discuss.huggingface.co/t/gemma-4-bug-fixes-and-research-request/176979
- Papers (small models 0.6B-8B; none is Gemma 4 12B):
  - Outlines/XGrammar took schema validity from 78.6-92.9% to 100% on 0.6-4B models, but
    multi-step / instruction-semantic failures remained. https://arxiv.org/abs/2609.23742
  - StructureBench (IJCAI 2026, 11 on-device models): enforces validity, does not reliably improve
    semantic accuracy, can hurt small models or complex grammars.
    https://www.ijcai.org/proceedings/2026/0267.pdf
  - "Brief Is Better" (Qwen2.5-1.5B, BFCL v3): constrained decoding removes hallucinated function
    names, but wrong-but-valid picks stay (24.5% at 1.5B, 9.5% at 7B with no reasoning).
    https://arxiv.org/abs/2604.02155
  - JSON-format benchmark: grammar was a guardrail for weak models (a 7B coder 0% -> 75%), but strict
    grammar hurt a very large model one-shot. https://arxiv.org/pdf/2603.03306
- Speed: grammar sampling costs CPU time per token. No Gemma 4 number found; its vocabulary is large,
  so measure it.

## 3. Fit for Grace
The 8-call loops with bad arguments likely come from the meta-tool design: `use_tool` takes
`{name: string, args: object}` with `args` a free object (`services/tools_server.py` ~l.868). A grammar
built from that tool forces valid JSON but cannot force `args` to match the chosen inner tool, since
the inner schemas are not in the request (that is why they are not: 130 schemas would blow the
cached prefix). Routes to constrain args to the chosen tool:
1. Per-call custom grammar: after `find_tools` / suggestions, send the next request with a
   `grammar` or `json_schema` that is a `oneOf` over only the 1-3 suggested tools
   (`{name: const, args: <that tool's schema>}`). The grammar is not part of the prompt, so the
   cached prefix is unaffected. Open question: whether llama-server lets a request `grammar`
   coexist with `tools` + `--jinja` (I believe the tool grammar takes over; not confirmed).
   Fallback: for that step, do not use native tool calling, have Gemma emit a plain JSON object
   under `response_format` (Grace-owned protocol) and let the engine run it.
2. Expose the suggested tools as real `tools` for that request. In Gemma's template tools render in
   the system turn at the start, so this breaks the prefix cache. Only viable on its own slot
   (`-np 2`) with its own stable prefix.
3. No decoding change: validate args in `tools_server.py` before running and return the expected
   schema in the error (WORKING_NOTES already guesses this). Evidence supports describing the failure
   rather than echoing the failed call back: https://arxiv.org/abs/2608.23651 (135M-1.7B models, so
   weak transfer).
Windows/CUDA: no issue, sampling is CPU-side. Danish in / English out: unaffected; free-text args
(search queries) stay plain strings.

### Gemma 4's non-JSON native format and the fallback parser (from R0)
- Gemma 4 writes `<|tool_call>call:name{key:<|"|>text<|"|>,...}<tool_call|>`, not JSON. llama.cpp must
  parse that (PEG-based parser; if the template is unrecognised it falls to `Generic`, which prompts
  for JSON instead and uses more tokens). A llama.cpp warning about an "outdated gemma4 template"
  appears when the GGUF's embedded template is old (seen in a community template repo,
  https://github.com/asf0/gemma4_jinja, third party). Check our Google QAT GGUF's template version and the
  startup log line.
- Reported failures (single-author claims, not confirmed): calls degrade to plain text in `content` under
  long system prompts, many tools and 3-4 sequential calls (issue #22786 and the HF thread above;
  vLLM also ships a fallback layer for this, https://docs.vllm.ai/en/v0.19.1/api/vllm/tool_parsers/gemma4_tool_parser/).
  This matches part of our failure set: "answers without a tool" and "promises then does nothing"
  (tools.rs already re-prompts "Nothing has been done yet"), so some of these may be parse misses, not
  model choice. Cheap check: log turns where `content` contains `call:` or `<|tool_call>`.
- Levers: (1) tolerant engine-side fallback parser for call text found in `content` or
  `reasoning_content`; (2) grammar-constrained retry of just the call (route 1 above) when the parse
  fails; (3) keep the prompt short, which also helps latency. No source found on lazy grammars for
  Gemma 4 specifically; test directly (step 1 of the test).

## 4. Expected gain and risks
- Gain: eliminates malformed / hallucinated-name / wrong-type calls. It will NOT fix near-miss tool
  picks or answering without a tool; the papers agree. Whether the 8-loops are syntax errors or
  right-shape-wrong-value is unknown; the traces log args and ok for every call, so check first.
  If mostly wrong values, grammar gives little.
- Cost: a 130-way `oneOf` is large, so restrict to the suggested few. Unmeasured per-token CPU
  overhead; possible quality dip from forcing format. Schemas exist in the registry already, so
  generating grammars is small work.
- Risk: grammar/tool/reasoning-sampler behaviour changes between llama.cpp builds; pin b11539 and
  retest on upgrade.

## 5. How we would test
1. Read b11539 `common/chat.cpp` for the Gemma 4 path: grammar set or not, lazy or not.
2. Classify the bad 8-loops in existing traces (f01, f07, f10, sys11): schema error / wrong value /
   right call but unhelpful result.
3. A/B on the 162-scenario tools sim (after fixing the stale `tools:` lists): baseline vs (a)
   pre-validation + schema-in-error vs (b) per-call `oneOf` grammar over suggested tools. Metrics:
   pass rate, `use_tool` calls per reply, replies hitting 8, time to first tool call p50/p95.
   Adopt (b) only if it halves 8-loops beyond (a) and adds < 30 ms to the first call.

## 6. Verdict
maybe: do (a) pre-validation + schema-in-error first (free); a per-call grammar over the suggested tools
is worth a spike only if traces show malformed args, since the literature says grammars fix syntax, not tool choice.
