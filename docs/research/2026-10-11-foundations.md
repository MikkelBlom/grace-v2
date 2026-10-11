# Research: memory, listening, fine-tuning (2026-10-11)

Prompted by Mikkel's question whether the fixes after the live memory sessions treated causes or
symptoms. Three literature reviews; vendor numbers and 2026 preprints are indicative, not settled.

## Memory

What the evidence says:
- **Updates are everyone's weak point.** HaluMem scores memory operations directly: no system
  got more than 47 % of required updates (Zep best; Mem0 and others missed 74–98 %), and extraction
  errors carry into answers. https://arxiv.org/html/2511.03506v2
- **Raw turns beat extracted facts as the stored unit.** LongMemEval: store one *round* (user turn
  plus reply); facts/summaries *instead of* rounds lose information; facts added to the *search
  key* while returning the raw round give +9.4 % recall, +5.4 % accuracy. Small readers degrade
  above ~3k tokens of retrieved text. https://arxiv.org/html/2410.10813 — On LoCoMo, full context
  (72.9) beat Mem0 (66.9) and Zep (66.0); Letta's grep over transcripts scored 74 %.
  https://arxiv.org/html/2504.19413, https://www.letta.com/blog/benchmarking-ai-agent-memory
- **Invalidate, don't delete; bitemporal with provenance** (Zep/Graphiti).
  https://arxiv.org/abs/2501.13956 — Mem0 now only adds facts and settles which is current at
  read time by date. https://docs.mem0.ai/core-concepts/memory-evaluation
- **Self-reinforcement is documented:** agents copy what they retrieve, and errors written back
  from their own output amplify; strict add criteria + deleting bad records ≈ +10 %.
  https://arxiv.org/abs/2505.16067
- **Prospective memory needs a typed store run by code.** PM-Bench: best agent 65 % F1.
  https://arxiv.org/abs/2607.12385 — A code-managed intention store took a small Gemma from 4.2 %
  to 66.2 %; seven ordinary memory approaches stayed ≤ 6.6 %. https://arxiv.org/abs/2609.01272
- **Small extractors:** Memory-R1 RL-tuned 3B–14B models on 152 QA pairs for ADD/UPDATE/DELETE/
  NOOP and beat baselines. https://arxiv.org/abs/2508.19828 — Small extractors often emit broken
  UPDATE/DELETE JSON that fails silently. https://arxiv.org/pdf/2609.10413

Our state, judged: provenance (user words only), learn-on-interrupt and bitemporal facts are
principled. The "no longer true" prompt list, followup-as-fact-kind, embedding-distance dedupe
(should flag, not decide) and the garbled/momentary prompt rules are symptom patches.

Recommended direction (not yet decided, see WORKING_NOTES):
1. Episode log (rounds, verbatim, dated, with STT confidence and interrupted flag) is the source
   of truth; recall returns dated rounds, indexed with their facts in the key.
2. Facts are a derived, bitemporal cache citing user turns; prefer add over update when unsure;
   newest dated wins at read; session-end reconciliation pass with the whole session in view.
3. Summaries never authoritative: render the profile block from facts by code, or regenerate.
4. Commitments table (what, trigger, status, source turn), surfaced by code at session start.
5. Gates: STT confidence, scope (session vs permanent), duplicate flag.
6. Memory regression benchmark in `grace-bench`: YAML multi-session scripts with a fake clock,
   deterministic DB assertions per operation + probe questions; categories from LongMemEval plus
   our live failures. 30–50 scenarios to start.

## Listening (STT)

- **We measured on the wrong speech.** RyeAI leaderboard: edda-v0.2 7.3 % WER on FLEURS but
  **15.6 % on CoRal conversation** — still the best open model (brage 16.2, hviske-v6 18.3, ekko
  24.2, Parakeet-v3 50.3, Canary 46.4). https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard
- **AEC nonlinear processing hurts ASR** (residual echo suppression, comfort noise): artifacts,
  not leftover noise, cost WER; mixing raw back in helps. https://arxiv.org/abs/2201.06685,
  https://arxiv.org/html/2404.14860v1 — Use the linear AEC output for STT, suppressed output for
  VAD/barge-in.
- **Whisper invents text from non-speech** (~0.6 % of CoRal conversation for large-v3; more around
  long pauses). https://arxiv.org/abs/2402.08021 — Proven fixes: VAD-trim the segment, no
  temperature fallback, no conditioning on previous text, train with non-speech → empty target.
  Calm-Whisper: fine-tune 3 decoder heads, >80 % fewer hallucinations, <0.1 % WER cost.
  https://arxiv.org/abs/2505.12969 — Transducer/CTC models hallucinate far less.
- edda's numbers use fp16 beam 5; the CTranslate2 conversion is untested by its authors; we run int8.
- Smart Turn v3 on Danish: 93.1 % accuracy, 4.9 % false "turn ended", worst of the European
  languages. https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- Our avg_logprob −0.55 cutoff was calibrated on read speech; likely drops real casual speech.

Ranked hypotheses for live misrecognitions, each testable on `traces/audio/` turns: casual speech
is just harder (needs a labelled baseline); AEC suppression damages speech (decode raw vs AEC on
turns where Grace was silent); non-speech reaches Whisper (speech share per turn); decoding
settings (int8/greedy/fallback vs fp16 beam 5 temp 0); early cut-offs; level/clipping/aliasing.

## Fine-tuning

Feasible on the RTX 5090 Laptop (PyTorch cu128 supports sm_120; Unsloth on Windows/RTX 50).
Ranked by value / effort:
1. **Smart Turn v3 on Danish** — 8M params, BSD-2 training code, CC-BY v3.2 dataset (271k clips).
   Danish data: forced-aligned cuts of Common Voice / FLEURS / NST, CoRal (gated), synthetic Danish
   TTS, and weak labels from Grace's own sessions (Mikkel continued after a pause → incomplete).
   Mix in the multilingual set to avoid forgetting. https://github.com/pipecat-ai/smart-turn
2. **STT LoRA** on Mikkel's own turns (30–60 min corrected) plus his room's non-speech with empty
   targets (Calm-Whisper style). 12–24 % relative WER gains reported for speaker LoRA.
   https://arxiv.org/pdf/2408.03979 — Unknown: conversion back to CTranslate2.
3. **LLM LoRA** (Gemma 4 12B fits for LoRA in 24 GB via Unsloth): only for memory extraction, and
   only if grammar-constrained prompting fails on a labelled set. Persona fine-tune: unproven for
   short turns, risks tool use. Training data from Claude: Anthropic's Commercial Terms bar
   training "competing AI models" — ambiguous for a personal LoRA; use a local teacher instead.
4. **Danish TTS needs no training:** CoRal Røst v3 Chatterbox (350M real-time / 500M quality),
   MOS 4.23, zero-shot cloning. https://huggingface.co/CoRal-project/roest-v3-chatterbox-350m —
   bake-off for latency.
5. VAD/AEC: don't train.

## Personality source

v1's spec is `config/personality.json` in the v1 repo (not HTML): witty *and warm*, dry quick
humour, sharp but never arrogant, "a sharp friend, not a manual", honest pushback done kindly,
vary wording, match his tone, infer misheard words, ask "did you mean…?" before acting on
nonsense, `<SKIP>` for pure backchannel, plus examples.
