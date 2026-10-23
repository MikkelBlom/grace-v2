# RL fine-tuning (GRPO) for tool use with a simulator reward

Desk research, 2026-10-23. Nothing was run.

## 1. What it is
Instead of showing the model correct answers (SFT), let it try a task several times, score each attempt
with a program (did it call the right tool with the right arguments, did it answer), and push up the
probability of the better attempts. GRPO does this by comparing a group of samples for the same prompt, no
separate value model. "Verifiable rewards" means the score comes from a checker, not a judge model.

## 2. Evidence
- **ToolRL** (Qian et al., NeurIPS 2025, [arXiv 2504.13958](https://arxiv.org/abs/2504.13958)): GRPO with a format reward plus
  decomposed correctness (tool name, parameter names, parameter values). Reported about +17% over base models and +15% over SFT models
  (abstract). Qwen2.5-3B example from the paper's table: Original-reward 52.98% vs coarse 51.40%; for 1.5B 46.20% vs 36.72%
  (via search snippet of the paper; verify before citing). Code/data: [github.com/qiancheng0/ToolRL](https://github.com/qiancheng0/ToolRL).
  Finding: finer-grained rewards beat coarse ones; length/reasoning-length rewards hurt (secondary summary, unverified).
- **Nemotron-Research-Tool-N1** (NVIDIA, ICLR 2026, [arXiv 2505.00024](https://arxiv.org/abs/2505.00024)): Qwen2.5-7B/14B trained with
  GRPO and a *binary* reward on format + correct call. Paper claims the 7B/14B beat GPT-4o and xLAM-2-70B / ToolACE-8B on BFCL;
  secondary summary gives 84.82% vs 83.97% (unverified against the table). Ablation (secondary): strict binary beat partial credit on
  "Live" data (80.38 vs 76.61). Note the opposite lesson to ToolRL: reward shape is task-dependent, so try both.
- **Fission-GRPO** (Zhang et al., ACL 2026, [arXiv 2601.15625](https://arxiv.org/abs/2601.15625)): targets exactly our "loops of repeated
  bad calls" failure. Each failed rollout is "fissioned": the error is turned into a new on-policy training instance, with diagnostic
  feedback from a fine-tuned error simulator, and the model samples several recovery attempts. On BFCL v4 multi-turn with Qwen3-8B:
  +5.7 points absolute error-recovery rate and overall accuracy 42.75% to 46.75% (+4.0), better than plain GRPO (paper-reported; I read the abstract-level
  summary only). Fit: our tools server returns real error strings, so a templated or Gemma-written error simulator is possible; the
  repeated-call loop is directly scoreable in the sim.
- **ToolExpander** (OPPO/ECNU, [arXiv 2510.07737](https://arxiv.org/abs/2510.07737)): for weak small models where plain GRPO stalls or collapses
  (its Qwen2.5-1.5B plain GRPO run collapsed). Two tricks: dynamic multi-round hard sampling (prompts the model still fails after rollouts are replaced
  by few-shot demonstrations built from other samples using the same tools; replace beat add) and "self-exemplifying thinking" (drops the KL term,
  clip 0.26/0.2, tiny +0.01 bonus). Qwen2.5-7B BFCL 0.8248 base, 0.8452 GRPO, 0.8561 dynamic-replace, ~0.86 self-exemplifying; APIBank 62.66
  GRPO vs 81.76 dynamic-replace; ACEBench mixed. Needed 8x A800 80 GB with veRL, batch 1024, so it is not a 24 GB recipe as published.
  Authors flag reward hacking in self-exemplifying. Fit: Gemma 4 12B is not a "weak" model, so likely much less needed; the hard-sample
  idea (swap in demonstrations for scenarios that never succeed) is cheap to copy. Dropping KL is risky for persona/Danish drift.
- **Single-GPU feasibility**: TRL GRPO has a colocated generation mode for one GPU; rule of thumb sources say GRPO needs 2-3x SFT VRAM and
  >=8 samples per prompt ([TRL runbook-style guides](https://huggingface.co/Paerle/guido-grpo-runbook), weak sources). Unsloth claims
  ~90% less GRPO VRAM and that 24 GB handles GRPO for models up to ~24B (stated for Gemma 3 12B era; [Gemma 3 post](https://unsloth.ai/blog/gemma3),
  [long-context GRPO](https://unsloth.ai/docs/jp/meru/reinforcement-learning-rl-guide/grpo-long-context)). Unsloth's Gemma 4 page states
  GRPO works with `fast_inference=False` (no vLLM) and E2B GRPO takes 9 GB ([guide](https://unsloth.ai/docs/models/gemma-4/train)); no 12B figure.
  Without vLLM, rollout generation is slow (HF generate), which is the real bottleneck, vendor claims aside.
- Most published results are Qwen2.5 1.5B-14B, short single-turn BFCL-style calls with in-distribution training sets. Multi-step agentic RL
  with live environments (Agent-R1, rLLM-style, "SWE" agents) usually uses multi-GPU clusters; I found no 24 GB laptop result for a 12B Gemma.
  Gemma 4 + GRPO tool-use papers: none found.

## 3. Fit for Grace
- Hardware: QLoRA GRPO on 12B probably fits 24 GB with short completions (tool calls are short) and a trimmed prompt; our prompt prefix
  (system + tool list) makes every rollout long, so prefill dominates. Unverified; test memory first. llama.cpp is serving-only and cannot
  roll out inside the trainer; rollouts would go through HF/Unsloth, so sampling behaviour may differ subtly from llama.cpp Q4_0.
- Reward source: grace-tools-sim already checks tools called (in order), arguments and reply regexes per turn against a real llama-server and
  tools server in `--dry-run`. For RL it needs: (a) an in-process/headless scorer that takes a rollout (list of tool calls + text) without
  the full engine round trip, (b) the tools server called with `--dry-run` for side-effect-free execution, (c) per-turn partial credit
  (right tool 0.3, right args 0.4, order, no extra calls, English reply, no-tool correct, penalise repeated calls). A rollout that needs a
  multi-step conversation means the environment is stateful: a few hundred ms to seconds per tool call (web/file search "run for real" in
  the sim; they must be mocked or recorded for RL, `fixtures/tools/recorded` helps).
- Reward hacking risk: regex checks on replies are gameable; the model learns to emit what passes. Held-out scenario families are mandatory.
- Prompt cache/latency: unchanged at serving time if the adapter is applied the same way as in fine-tuning.md.

## 4. Expected gain and risk
In papers, RL beats SFT on generalisation to unseen tools and avoids SFT's memorisation, which suits our loops and near-miss failures,
and needs only prompts + scorer (no labelled answers, clean licensing). Cost: rollouts per step = prompts x group size (>=8), each
with a long prefix; compute grows quickly and I cannot give a measured range for a 12B on this GPU. Risks: reward hacking, drift of persona and
Danish (a KL penalty / LoRA with low LR helps), more engineering than SFT, rollout-vs-llama.cpp mismatch (quantization).

## 5. How we would test it
Stage 0: build the offline scorer and verify its rewards match grace-tools-sim verdicts on recorded runs (agreement >95%).
Stage 1 (cheap check): rejection-sampling SFT (training-data.md) first; RL only if that plateaus.
Stage 2: GRPO on a small model first (E4B or JevK5-class) to debug the reward, then 12B QLoRA with G=8 on single-turn scenarios, scoring
on held-out scenario families and a sealed set. Adopt if held-out sim gains clearly exceed RFT/SFT at similar compute, no reward-hacked reply
styles in transcript review, chat/style/Danish benches within noise.

## 6. Verdict
**maybe** (second step after SFT/rejection sampling): the reward infrastructure is mostly there, literature is encouraging on small models, but no 12B-Gemma-on-24 GB evidence and rollouts are slow and engineering-heavy.
