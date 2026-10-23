# Tiny dedicated tool routers (R1: tool-routing)

Desk research, 2026-10-23. Search snippets plus abstract fetches; full PDFs were not readable, so
numbers below are from abstracts/model cards. Flagged where vendor or single-author.

## 1. What it is
Grace picks hidden tools with a second `llama-server` request (`Suggester` in
`services/tools_server.py`): Gemma reads a one-line catalog (~130 tools), names up to 3. It scored
54/56 at 129 ms p50 on our scenario lines. harrier-270m embedding retrieval found the right tool in
the top 4 only 27/56. The question: can a tiny dedicated model (function-calling 270M, reranker,
BM25 hybrid) take over this job, off the main Gemma slot, on CPU or the Arc iGPU?

## 2. Evidence
- **FunctionGemma 270M** (Gemma 3 270M tuned for function calling). Google's own numbers: Mobile
  Actions eval 58% base -> 85% after fine-tuning (vendor claim). Zero-shot BFCL per the model card as
  summarised by DataLearner: Simple 61.6, Parallel 63.5, Multiple 39.0, Parallel-Multiple 29.5. The
  Multiple score (choose among several tools) is the one that matters for us, and it is weak. No
  published result on a ~130-tool catalog. A community Arabic fine-tune eval of the base model shows 0.0
  tool-name accuracy, i.e. results swing hard with domain/language. Google positions it as a
  router for simple commands that hands hard ones to a bigger model, which is our use, but gives no
  latency numbers. Sources: https://blog.google/technology/developers/functiongemma/ ,
  https://deepmind.google/models/gemma/functiongemma/ ,
  https://www.datalearner.com/ai-models/pretrained-models/functiongemma-270m-it ,
  https://huggingface.co/metga97/functiongemma-270m-ar-tooluse/blob/main/eval_report.json . An Ertas blog
  claims 82-88% with no method; ignore.
- **Lexical + small neural, closest to our case** (arXiv 2609.18672, one author, on-device routing, 70
  actions, 600 requests): character 3-gram BM25 is right on 162/164 requests that reuse catalog words but
  only 85/166 paraphrases. Narrowing candidates to 7 lifts paraphrase accuracy to 0.825. A Qwen3-4B
  choosing from a BM25 top-5 reached 0.777; e5 reranking of the BM25 top-5 0.642. Conclusion: selection
  is mostly retrieval, but paraphrase is the hard part. https://arxiv.org/abs/2609.18672
- **Tuned retrievers do not fix recall** (arXiv 2608.22695): a domain-tuned tool retriever scored recall
  0.608/0.596 vs 0.625/0.626 for general encoders; first-stage recall stays the bottleneck.
  https://arxiv.org/pdf/2608.22695 . ToolRerank (2403.06551) shows a tool-specific reranker helps over plain
  retrieval, 2024, older benchmarks. https://arxiv.org/pdf/2403.06551
- **Generic hybrid retrieval** (dense + BM25, then cross-encoder) is the 2026 default in practitioner write-ups;
  small rerankers (tens of millions of params) are CPU-cheap. No tool-specific numbers; blog-level evidence.
- Already in the R0 sweep: tool-retrieval papers report 97% hit@3 on self-built sets (2603.20313). Our own
  27/56 shows the Danish, spoken, near-miss-heavy reality is much harder than those sets.

## 3. Fit for Grace
- A 270M model is trivial in VRAM/RAM. FunctionGemma is a Gemma 3 architecture, so llama.cpp runs it (GGUF
  exists); it has its own call syntax and is meant to be fine-tuned. Danish input: Gemma's 256k vocab
  helps, but base quality in Danish is unproven; fine-tuning data would be programmatic (templates +
  paraphrases, our tool descriptions), which is within the training-data rules. Check the Gemma terms/licence
  before shipping a fine-tune (the blog does not state it).
- CPU/iGPU: 270M on CPU is tens of ms for a short prompt; the prompt is the problem. A 130-tool catalog is
  ~3-5k tokens; a fine-tuned router would instead bake the catalog into weights (classify to a tool id) and
  take only the utterance, which is the efficient design. Then prefill is ~50 tokens and p50 should land
  well under the 129 ms we have (expectation, not measured).
- It removes the second Gemma request, but the current request is already cheap (prefix cached, 24
  max tokens, 129 ms). The gain is slot contention (`-np 2`), not the latency itself, plus isolation from the
  main prompt cache (note R0: the cache thrash risk).
- Multi-label: we need "top 3 or none". A classifier head or a constrained generate over tool ids handles this;
  abstention (chat vs tool) is the genuinely hard part per 2609.18672 (BM25 AUC <= 0.70, e5 0.81).
- Catalog changes (new tools) would need retraining or a retrieval front, a maintenance cost the Gemma
  catalog request does not have. That is the real trade: 54/56 with zero training vs. a tiny model that must
  be retrained when the registry changes.

## 4. Expected gain and costs
- Latency: little (129 ms -> maybe 20-60 ms). Value is freeing a Gemma slot and removing a dependency
  on the main server for routing, and in principle starting the route on partial transcripts.
- Accuracy: unlikely to beat 54/56; plausible target 48-52/56 after fine-tuning on synthetic paraphrases.
  Zero-shot FunctionGemma will not get there (BFCL Multiple 39).
- Costs: training data generator, fine-tune (Unsloth/NeMo recipes exist), retrain on registry change, an
  eval harness separate from tools-sim; held-out phrasing must be real-ish, and Mikkel is not the test set.
- Cheaper intermediate: BM25/char-3-gram on tool names + descriptions (+ Danish aliases) fused with harrier,
  as a candidate narrower for Gemma, or as a cheap "is a tool even needed" prefilter.

## 5. How we would test
1. Offline, no GPU needed for the first step: score BM25/char-3-gram and RRF(BM25, harrier) on the 56 lines
   and on a larger synthetic paraphrase set; report top-1/top-3/top-4 and abstention AUC.
2. Zero-shot FunctionGemma 270M and a 0.6-1B small LLM with the full catalog on the same 56 lines (CPU),
   accuracy + p50 latency.
3. Fine-tune FunctionGemma (or a 270M-class encoder + head) on programmatic data (templates, paraphrases by an
   open-weight teacher with permissive licence, self-distillation from Gemma's own 54/56 picks, which is
   allowed); evaluate on a held-out set written separately from training templates.
Adopt if: >= 52/56 top-3 incl. abstention correct, p50 <= 60 ms on CPU or Arc, and the registry-change retrain
is scripted. Otherwise keep the Gemma catalog request.

## 6. Verdict
**Maybe, low priority.** Zero-shot FunctionGemma is too weak (BFCL Multiple 39); a fine-tuned router could match
the Gemma catalog only with real training effort and a retrain-on-registry-change burden, for a small latency
gain. Do the cheap step first (BM25 + harrier fusion, and a tiny abstention gate) and keep the Gemma catalog
request as the default.
