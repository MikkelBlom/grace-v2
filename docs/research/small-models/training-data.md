# Training data for tool use, persona and Danish input

Desk research, 2026-10-23. Not legal advice. Re-read the terms before relying on them; they change.

## 1. What it is
Where to get fine-tuning (or RL prompt) data without using Claude or Gemini output and without
making Mikkel the data source: write it by program, let Gemma generate and a bench filter it, or
use open-weight teachers whose licence allows it.

## 2. Evidence

### The terms that rule out Claude and Gemini (quoted)
- Anthropic **Consumer Terms**, Section 3 (Use of our Services): you may not use the Services "To develop any
  products or services that compete with our Services, including to develop or train any artificial intelligence
  or machine learning algorithms or models or resell the Services."
  ([consumer terms](https://www.anthropic.com/legal/consumer-terms)). Claude Code on a Pro/Max plan falls under these.
- Anthropic **Commercial Terms** (API, Team/Enterprise) bar using the Services "to build a competing product or service,
  including to train competing AI models" unless Anthropic approves ([commercial terms](https://www.anthropic.com/legal/commercial-terms)).
  Wording seen via search snippets; check the live page. Same effect: no.
- Google **Gemini API Additional Terms**: "You may not use the Services to develop models that compete with the Services
  (e.g., Gemini API or Google AI Studio)." ([terms](https://ai.google.dev/gemini-api/terms)). The consumer Gemini app has its own terms
  (not fetched); assume the same direction.
- Practical note: the rule is about *our* model being trained on their output. A local Gemma 4 judging or generating is fine. Do not
  use Claude/Gemini to write, label, grade or "clean" training rows, including this repo's agent sessions writing scenario text that
  becomes training data in bulk. Hand-writing templates by an agent is a grey zone; keep templates short and human-reviewed, and
  prefer programmatic variation. Flag to Mikkel if he wants a clear line.

### Open-weight teachers and licences
- **Gemma 4 (all sizes incl. 26B-A4B, 31B): Apache 2.0**, text of [the licence page](https://ai.google.dev/gemma/docs/gemma_4_license)
  is plain Apache 2.0 with no output or "train other models" clause (earlier Gemma 1-3 used the custom Gemma Terms with a
  Prohibited Use Policy). So Gemma 4 31B as an offline teacher for Gemma 4 12B is clean as far as I can read.
  I did not check the separate "Terms of use" link on that page; do that once.
- **Qwen3 / Qwen3.5 / Qwen3.6**: Apache 2.0 for the official open checkpoints per secondary sources; I could not reach the primary
  licence files. Check each repo's `LICENSE` (larger or special variants have differed in the past). Not a Llama-style
  "no training other LLMs" clause as far as I know, unverified.
- Avoid: Llama community licence (restricts using outputs to improve other LLMs; naming rules), and any model distilled from
  Claude/Gemini ("Opus-distilled" community checkpoints): their provenance taints the output.

### Self-distillation / rejection sampling
- STaR (Zelikman 2022, [arXiv 2203.14465](https://arxiv.org/abs/2203.14465)) and RFT / rejection-sampling fine-tuning
  (Yuan 2023, [arXiv 2308.01825](https://arxiv.org/abs/2308.01825)): sample N answers, keep those a verifier accepts, fine-tune on
  them. RFT reported gains growing with number of distinct correct paths. Works when the verifier is reliable; our sim is a
  verifier for tool name/args order and reply regexes.
- ToolACE's data pipeline: synthesize APIs, simulate dialogues with multiple agents, then dual-layer verification (rule +
  model); 26,507 APIs ([paper](https://arxiv.org/abs/2409.00920)). Hammer adds *function masking* (rename functions/arguments so the
  model reads descriptions instead of memorising names) and irrelevance data (7.5k samples) - both relevant to near-miss picks.

- **CacheRL warning** (Islam et al., June 2026, [arXiv 2606.14179](https://arxiv.org/abs/2606.14179), section 6.3; Qwen3-4B-Thinking,
  author results): rejection-sampled data kept only the best-scoring answers and these trajectories were short (673 samples averaging
  1.5 tool calls vs 8.1 in the original SFT data). Training on the curated samples alone cut the tool-calling rate from 83% to 52%.
  Mixing the curated samples with the original data restored tool-call diversity and kept the answer-quality gain (validation reward
  0.432 to 0.607 for original + curated). Their ablation also shows the unfiltered 3,689-sample subset alone was weaker, and that
  combining both beat either (0.43 to 0.76). Lesson for us: our sim filter must not select for "answers without a tool" or
  fewer calls by accident (it would also reward the very failure we want to remove). Always mix filtered rows with the original/templated set,
  keep the tool-call rate per scenario type as a monitored metric, and check it on the held-out split. Same paper: without a minimum-token
  constraint 42% of RL rollouts at step 0 made no tool call (13% with it at step 0 only).

### Size and diversity (what published work used)
- Hammer: ~60k APIGen samples + 7.5k irrelevance. xLAM/APIGen: 60k verified. ActionStudio: 98k trajectories
  ([arXiv 2503.22673](https://arxiv.org/abs/2503.22673)). Those are generalists across thousands of APIs. For a fixed ~130-tool
  catalog, far fewer is a reasonable expectation (low thousands), but I have no paper measuring that threshold: this is judgement.

## 3. Fit for Grace
Our assets: ~162 sim scenarios (many stale), the 130-tool registry with schemas, `fixtures/tools/recorded`, and the sim as a
verifier. Danish STT-style inputs: generate by template + noise (dropped words, STT-typical misspellings, code-switching), and
from public Danish corpora (e.g. Danish Gigaword / CoNLL-style text, public NST/CommonVoice transcripts: check licences) used as
*utterance* material, not Mikkel's voice. Persona: general rules from style work, not input->reply pairs in the prompt (memory note);
for training, persona examples would be generated by Gemma under the persona prompt then filtered by grace-style-bench.

## 4. Expected gain and risks
- Programmatic templates: perfect labels, low diversity, risk of learning the template. Mitigate with many phrasings, argument
  distributions, distractor tools, no-tool cases, multi-turn and error-result cases.
- Self-distillation: only teaches what the model sometimes already does; fixes consistency (loops, no-tool answers) more than
  capability. Cheap, clean licence.
- 27-31B teacher: could add capability; slow (31B Q4 ~17.5 GB, runs on this GPU offline at a few tok/s to ~25 tok/s; unmeasured here);
  label noise needs the sim filter.
- Risk: overfitting the sim. Splits below.

## 5. How we would test it
- **Splits by construction, not by row**: split scenario *families* (tool group / intent template), not individual lines. Hold out
  e.g. 25-30% of tool groups entirely (unseen tools -> tests generalisation of the catalog skill) plus a held-out template-free set.
- **Never train on** the final eval scenarios; keep a frozen "dev-sim" for iteration and a separate sealed "test-sim", written by a
  different generator or hand, run rarely. Add paraphrase/noise perturbations of eval lines to detect memorisation.
- **Also gate on** grace-chat-bench, grace-style-bench, and a Danish-comprehension set (public corpus), so we notice forgetting.
- Experiment: generate ~2k rows three ways (templates / Gemma rejection sampling N=8 / Gemma 31B teacher), fine-tune the same recipe on each
  (see fine-tuning.md), compare on sealed test-sim. Adopt a source if it beats baseline on sealed test by a margin over seed noise
  and does not lose on chat/style/Danish.

## 6. Verdict
**adopt-candidate** for programmatic generation + self-distillation with sim scoring (clean licence, no cloud, no Mikkel); **maybe** for a Gemma 4 31B teacher (Apache 2.0 looks clean; confirm the Terms-of-use page). Claude/Gemini-generated data: **not allowed**.
