# Scaffolding: agent designs that make small models reliable at tools

Desk research 2026-10-23. Nothing was run. Evidence is thin for 12B-class models on tool tasks;
most papers use 0.1B-8B models or large models on tau-bench, so every gain below is a hypothesis for
Gemma 4 12B until grace-tools-sim says otherwise.

## 1. What it is
Code around the model that catches or prevents its mistakes: how errors are fed back, loop breaking,
pre-validation, a finish gate, a checker model, planner/executor splits, retrieval of tools, few-shot
examples. Grace already has: tool retrieval (Gemma picks from a one-line catalog, Hit@3 54/56 at 129 ms),
a hard cap of 8 calls (`MAX_CALLS`), a "Nothing has been done yet" re-prompt, and a 4B yes/no side model.

## 2. Evidence, by technique

**Error feedback format (primary source found).** "Feedback That Backfires: Why Small Language Model
Agents Repeat the Call They Just Watched Fail", arXiv 2608.23651. Six instruction-tuned models, 135M-1.7B,
4 families, simulated tool calling and MBPP repair: putting the failed call back in context made repeating
it MORE likely in all 6 (probability of re-emitting it on a fixed candidate set 0.06 -> 0.54; greedy decoding
copied the failed call token for token on 19% of items after a failure vs 0% before). About 83% of the
damage came from the failed call's surface form, not from the word "failed". Best fix: replace the verbatim
call with a runtime-written description of the failure (removes 76% of the effect, no extra tokens);
decoder-level blocking of failed strings attacks the same term; a "do not repeat" instruction did not change it;
retrying from a clean context (delete the failed attempt) was the worst. https://arxiv.org/abs/2608.23651
Caveat: models far smaller than 12B; direction likely transfers, size of effect does not. This matches our
loops: tools.rs appends the assistant message with the raw `arguments` back into history for every call.
Secondary blog that R0 first found: https://www.beri.net/article/small-model-agent-repeats-failed-tool-call-harness-error-format

**Hard stop on identical tool + args.** Repeated identical calls are a documented failure in small models
(ToolScan found Llama 3 8B and R1-distill-7B repeat calls even when told not to:
https://arxiv.org/pdf/2411.13547). No paper measures the stop itself; it is plain engineering: key on
(tool, canonical args), on the second identical call return a synthetic result ("already tried with those
arguments: <what happened>; try different arguments or tell him it didn't work") instead of running, and on
the third end tool use for the reply. Zero model cost.

**Argument pre-validation.** Check args against the tool's schema before running and return a description of
what was wrong plus the expected shape (not the failed call). Backed by the previous point and by the
constrained-decoding finding that schema errors are fixable but semantic ones are not
(https://arxiv.org/abs/2609.23742). Zero model cost; the registry has the schemas.

**G-STEP (arXiv 2605.00136).** R0 described it as a "finish gate". What I could verify: the paper ("Are Tools All
We Need? Unveiling the Tool-Use Tax in LLM Agents") says the tool-calling protocol itself costs accuracy under
semantic distractors and that G-STEP is a "lightweight inference-time gate" giving only PARTIAL recovery; the
abstract does not describe the mechanism, models or numbers, and I did not read the full paper. So the "finish
gate" reading is unconfirmed. A finish-gate idea with a clearer source: StateAct (arXiv 2607.22798) uses an
independent check when the agent calls finish (computer-use agents, not verified in detail).
https://arxiv.org/abs/2605.00136 , https://arxiv.org/pdf/2607.22798
For Grace the same idea is cheap to build ourselves: at the end of a reply with zero tool calls, if the suggestions
named a tool OR the reply promises an action, run the existing check (see verifier) before speaking. Note replies
stream sentence by sentence, so a gate after the answer is too late; it has to run BEFORE the first sentence is
emitted, i.e. on the hint path that already exists (the 4B "needs a lookup?" hint, 14/48 late in sim v2).

**AgentCheck (arXiv 2607.11098).** A fault-injection workbench for agents over MCP: records real tool responses,
replays with one response altered by one of 12 fault types (timeouts, stale data, poisoned descriptions, prompt
injection, ...). Five agents: best passed 105/120 scenarios, weakest 77; failures were mostly silent, confident use
of wrong tool output. Retry fixed timeouts for the weakest agent (as low as 30% -> 100%) but stale data stayed at
30-40%. https://arxiv.org/abs/2607.11098 . For us it is a TEST idea, not runtime: add a fault mode to the tools
server (error, empty result, timeout, stale, malformed args echo) and run the sim under it. Fits "not the test
set" (automated harness).

**ReAct vs function calling.** On tau-bench GPT-4o's function-calling agent beat Act and ReAct in the airline
domain (0.420 vs 0.365 / 0.325 pass@1) per one review; structured calls reduce format errors. Grace already uses
function calling; a ReAct text loop would add tokens and parsing. Not worth it
(https://themoonlight.io/fr/review/-bench-a-benchmark-for-tool-agent-user-interaction-in-real-world-domains,
secondary source).

**Planner / executor.** Evidence is one self-reported write-up (a planner at every user turn on tau-bench with a
30B model, 41.2% pass^1 on 177 tasks, no clean ablation):
https://medium.com/@duttasaswata7/structuring-multi-agent-systems-around-irreversible-actions-lessons-from-tau-bench-defe0f139eda .
A plan call before every tool turn is an extra full model round trip (hundreds of ms at 50 tok/s). Our
find_tools + suggestions step is already a planner for tool CHOICE. Skip.

**Verifier / critic (e.g. 4B checking the tool choice).** Verified-tool-call work shows external
postcondition checks help when a verifier is reliable (https://arxiv.org/pdf/2608.02645, simulator only).
Grace has JevK5 4B on the Arc iGPU at about 130-400 ms per decision, in parallel with the main model's prefill
if started early. A critic after the main model's tool call adds that on top of the call, before the tool runs
(not before speech if a filler line is spoken). Plausible use: yes/no "does this tool + args answer his
request" for non-destructive tools only when suggestions disagree with the pick. Unproven; the checker is itself
weak. Medium risk of false vetoes.

**Self-consistency / sampling several calls.** Needs N decodes: N x 50 tok/s at least for the call, or batch
across the two slots. Latency cost makes it a poor fit for the voice path; only for destructive tool calls,
where it can be a vote of 2-3 on the call JSON. Not evidence-backed here (no source read).

**Few-shot tool examples.** Small-model study: biggest jump at the first example (+8 points average, diminishing
after 3), task-dependent (Meta-Tool, https://arxiv.org/pdf/2604.20148); small models still struggle with JSON
format (https://arxiv.org/abs/2504.19277). Costs prompt tokens in the cached prefix. We can afford 1-2 compact
examples if the prefix stays fixed; we must keep the examples as principles, not input->reply pairs for the
persona (memory rule), but tool-call examples are format, not persona, so this is allowed.

**Tool retrieval / descriptions.** Already adopted. Tool-description rewriting (arXiv 2602.20426, Trace-Free+):
reports 60.89% better query-level success and 29% less degradation as the catalog grows past 150 tools, on
StableToolBench (agent model not stated in the abstract): https://arxiv.org/abs/2602.20426 . See
context-engineering.md.

## 3. Fit for Grace
All of the harness items (error rewrite, identical-call stop, pre-validation, fault injection) are Rust/Python
changes in `tools.rs` / `tools_server.py`, run on Windows, add no VRAM, no prompt tokens and no latency on
success paths. The prefix cache is untouched because only the tail of the conversation changes. Danish in /
English out is irrelevant to them. Critic/consistency/planner add model calls and are the expensive group.

## 4. Expected gain and risks
- Error rewrite + identical-call stop + pre-validation: should eliminate most of the 8-call loops (f01, f07,
  f10, sys11). Guess: loops to near zero; pass-rate effect depends on how many loop scenarios are recoverable.
  Risk: low. Stopping too early on tools that legitimately repeat (polling, "next page"): key on identical args
  only.
- Finish gate on the hint path: could address "answers without a tool" (c02, c04, f11, sys10); risk: false
  positives make Grace act on chit-chat; keep gate to cases where suggestions fired.
- Critic: unknown gain, +130-400 ms; only if loop fixes leave near-miss picks (sys07 volume -> media).

## 5. How we would test
Order matters, stop when sim stops improving:
1. Harness only: arm A = baseline; arm B = rewritten errors (no raw args echo) + identical-call stop +
   pre-validation with expected-schema message. Sim 162 x 3 runs after fixing the stale `tools:` lists.
   Metrics: pass rate, replies hitting 8 calls, mean calls per reply, p50/p95 time to final audio.
   Adopt B if loops drop >= 80% and pass rate does not fall.
2. Fault-injection mode on the tools server: run the sim under timeout / empty / malformed-args / stale faults,
   compare arms A and B; this finds silent-wrong-answer cases. Record the numbers like AgentCheck does.
3. Only then, a finish-gate arm and a 4B-critic arm on the residual failures; adopt if the net pass-rate gain
   exceeds 3 points per 100 ms added median latency.

## 6. Verdict
adopt-candidate: error rewrite, identical-call hard stop and argument pre-validation (free, no cache cost, backed by the
one small-model study that measured it); maybe: fault-injection harness and a finish gate; not for us: planner/executor,
ReAct, self-consistency on the voice path.
