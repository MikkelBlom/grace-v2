# Context engineering for a small tool-calling model

Desk research 2026-10-23. Nothing was run. Evidence on 12B-class models is thin; the strongest
sources are practitioner write-ups (Manus) and a few small-model papers.

## 1. What it is
How the prompt is written and ordered: tool descriptions, system prompt wording, where dynamic text
(memory, tool results, suggestions) goes relative to the cached prefix, how history is trimmed or
summarised, and whether the prompt is compressed. For Grace the prefix (system prompt about 9.6 KB, roughly
2.4k tokens, plus the tool list) is cached by llama-server per slot (`cache_prompt: true`, 16k per slot); a
changed token near the front costs a full re-evaluation, up to 1.5 s in live run 8.

## 2. Evidence
- **Cache discipline (Manus, practitioner write-up, hosted models).** Optimise KV-cache hit rate; keep the
  prefix stable (even one token changes everything after it; no timestamps at the start); make context
  append-only with deterministic serialisation; do not add/remove tools mid-task, mask instead; keep failed
  actions visible; beware few-shot ruts where the model copies repeated patterns; input:output is about
  100:1 so input efficiency dominates. Cached vs uncached price ratio 10x is for a hosted API; ours is
  latency, same logic. https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
- **Tool descriptions are a bottleneck.** arXiv 2602.20426 (Trace-Free+): descriptions written for humans are
  ambiguous for agents, the damage grows with catalog size; rewriting reported 60.89% better query-level
  success and 29.23% less degradation at 150+ tools on StableToolBench (agent model not stated in the abstract;
  vendor-free academic claim). https://arxiv.org/abs/2602.20426
- **Gemma 4 prompt format.** System turn exists; thinking and tools are declared in it; strings in calls
  use `<|"|>`; thoughts of earlier turns are dropped from history, kept within a tool-calling turn; "summarise
  earlier thoughts as plain text" is suggested for long agentic runs.
  https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
- **Few-shot:** first example gives the biggest jump (+8 points average on a small-model few-shot study, more than 3
  examples gives little); task dependent. https://arxiv.org/pdf/2604.20148
- **Format weakness:** small models struggle most with output format adherence.
  https://arxiv.org/abs/2504.19277
- **Error text in context:** raw failed calls raise the chance a small model repeats them; a written description
  of the failure helps (details in scaffolding.md). https://arxiv.org/abs/2608.23651
- **Prompt compression (LLMLingua / LLMLingua-2).** LLMLingua reports up to 20x compression with about 1.5 points
  loss on its benchmarks (https://arxiv.org/pdf/2310.05736); LLMLingua-2 is a token-classification model
  claimed 3-6x faster than LLMLingua (Microsoft repo, vendor claim: https://github.com/microsoft/LLMLingua).
  I found NO evidence on tool schemas, tool calling accuracy, or Gemma 4, and none on Danish. Compressed text
  also changes token sequences, so compressing the prefix can only be done offline and once; compressing it per
  turn kills the cache (my reasoning, not from a source).
- Lost-in-the-middle style position effects: not researched here; Gemma's chat format puts the system turn first,
  so recency favours text near the end of the latest user turn.

## 3. Fit for Grace
Already true: suggestions go into the user's message (after the cached prefix), tools picked from a one-line
catalog in the cached prefix. What the evidence suggests, in order of payoff:
1. **Keep volatile text out of the prefix.** The system prompt currently ends with the memory block ("Your memory (at
   the end of this prompt)"). Memory changes after every learned fact, so every change re-evaluates from that point.
   Fine if the memory block is the LAST part and the tool list sits before it; verify the ordering, and that the
   date/time is not rendered above the stable parts. Option: move memory into the first user message of the session,
   or accept the miss only when memory actually changed (measure `cache_n` per turn).
2. **Tool descriptions as the lever for near-miss picks** (sys07 volume -> media): add a 'Use for / Not for' clause
   to the ~15 confusable tools, in the catalog line (cached) and in the full description returned by `find_tools`
   (uncached tail). Rewrite from failures in the sim, not by guess. The Trace-Free+ paper suggests
   unambiguous, example-bearing, non-overlapping descriptions; training a rewriter is out of scope.
3. **Tool results trimmed and plain.** Cap result size, return the shape the answer needs, keep numbers verbatim
   (the calculator "heard" fix shows STT text must survive). Errors as descriptions of what went wrong plus the
   expected args.
4. **History:** per-turn `reasoning_content` is not stored; tool-loop messages appended as today. Summarise old
   turns into plain text before the 16k slot fills; Grace already has session summaries in memory. Do trimming at
   turn boundaries and in blocks (drop the oldest N turns at once), so the cache is invalidated rarely instead of
   every turn.
5. **Prompt length.** The persona section is long; the reply loop pays prefill only on a miss, but longer prompts
   mean longer misses and more distraction for a 12B. Test cutting persona prose that doesn't move the style
   bench (style bench decides, per the persona-principles memory rule); do not add input->reply lookup pairs.
6. **Compression:** not for the tool list or persona (cache, no tool-use evidence). Possible only for large
   tool RESULTS (web pages) before they enter the tail; but extractive truncation or a summary by the 4B is simpler
   and safer. Skip LLMLingua.

Windows/llama.cpp: all of the above is prompt text and ordering; `-np 2` shares the server with background jobs
(memory learning and style run on slot 2), so a second prefix variant competes for slot 2: consider that before
parking a thinking/other prefix there (see thinking-budget.md).

## 4. Expected gain and risks
- Prefix order / memory move: avoids the 0.5-1.5 s misses where they occur; range depends on the miss rate, which we
  can read from traces. Risk: memory in the user turn is less authoritative to the model than in the system turn.
- Description rewrite: improves tool picks on confusable pairs; low cost, effect only visible on sim scenarios.
  Risk: longer catalog lines lengthen the prefix (+tokens per tool x 130).
- History trimming: protects the 16k slot; low risk if done at turn boundaries.

## 5. How we would test
1. Measure first: from traces compute per-turn `cache_n/prompt_n` and which turns miss, and why (memory change,
   date line, summary change). Fix the largest cause.
2. Tools sim 162 scenarios x 3: arm A baseline; arm B with 'Not for' clauses on the confusable tools identified in the
   failure list; arm C B + 1-2 compact tool-call examples in the full description returned by find_tools (tail,
   not prefix). Metrics: pass rate, wrong-tool picks, tokens in the prefix, TTFT p50/p95. Adopt B/C if wrong-tool picks
   drop by a third without prefix growth above 5%.
3. Style bench / chat bench for any persona-prompt cut: adopt only if scores hold.

## 6. Verdict
adopt-candidate: measure and fix prefix cache misses (volatile text order), then rewrite confusable tool descriptions from sim failures;
not for us: prompt compression (LLMLingua) - no tool-use evidence and it fights the cache.
