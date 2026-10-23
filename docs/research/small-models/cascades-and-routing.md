# Cascades, routing and speaking before the LLM finishes

Desk research, 2026-10-23. Nothing was run. Thin evidence overall: most sources are vendor blogs;
the Grace-specific numbers come from `docs/MODELS.md`.

## 1. What it is
Two separate ideas:
- **Routing/cascade:** decide per turn whether a tiny fast model can answer ("hello", "next
  song", "volume down") or the 12B must. Either a router picks first, or the small model answers
  and escalates when unsure.
- **Early speech:** start talking (an acknowledgement, a filler, a canned phrase) while the real
  answer or a tool call is still being produced, so perceived latency drops without making the
  LLM faster.

## 2. Evidence

### How routers decide in practice
- RouteLLM (LMSYS/Berkeley, 2024): trained routers on Chatbot Arena preference data cut calls to
  the strong model by 35-85% (benchmark-dependent: >85% on MT-Bench, 45% on MMLU, 35% on GSM8K)
  while keeping 95% of GPT-4 quality; on out-of-distribution MMLU the arena-trained routers were
  near random ([LMSYS](https://lmsys.org/blog/2024-07-01-routellm/)). Lesson: the router must be
  trained or tested on your own traffic. These are text-quality results, not tool calling or
  voice.
- Voice-agent practice is mostly hand-written rules: LiveKit-oriented guides route on transcript
  length and "is it a question", and Retell-style flows give greeting/acknowledgement nodes a
  smaller model ([LiveKit guide](https://futureagi.com/blog/how-to-optimize-livekit-latency-2026/),
  [Retell guide](https://futureagi.com/blog/how-to-optimize-retell-latency-2026/)); LiveKit
  itself ships no router. A practitioner report routes deterministically (patterns first, model
  only on a miss) and gets p50 ~620 ms / p95 ~1.1 s
  ([write-up](https://www.craftedbydaniel.com/blog/my-voice-router-that-refuses-to-think-patternfirst-multiagent-orchestration-for-subsecond-latency)).
  A vendor claims 39% lower turn latency for local handling of simple turns
  ([slng](https://slng.ai/optimize-your-voice-agent)): marketing, not a benchmark.
- Another published pattern: classify intent on *partial* STT and prefetch the matching tool in
  parallel at confidence >= 0.85 (same LiveKit guide; example threshold, not a measured optimum).

### Small models that could serve
- **Gemma 4 E2B / E4B** (same family, Danish-capable, native function calling; Bedrock docs say
  E2B does one tool call per turn). Grace's own bake-off: E4B Tau2 42 vs 12B 69 (`MODELS.md`), so
  E4B is not a drop-in for tools but can be fine for chit-chat and fixed intents.
- **FunctionGemma 270M** (Gemma 3 base, 32K ctx): 58% on Google's Mobile Actions out of the box,
  85% after fine-tuning (secondary sources;
  [Google AI Edge post](https://developers.googleblog.com/on-device-function-calling-in-google-ai-edge-gallery/)).
  A narrow tool-picker/classifier, not a conversationalist; English-centric.
- **Qwen3.5-based JevK5 4B** already in Grace: 148/154 typed decisions, Danish ok, INT8 on the
  Arc iGPU.
- No Danish tool-calling benchmark for 1-4B models was found; BFCL scores for E2B/E4B were not
  retrievable.

### Where a side model can run
From `MODELS.md` (our own measurements):
| Placement | Cost to Gemma |
|---|---|
| RTX GGUF 4B alongside Gemma | Gemma decode 50 -> 33 tok/s while it runs; 133-208 ms per decision |
| Arc 140T iGPU, OpenVINO INT8 | Gemma decode 50 -> 48.5; 228-301 ms per decision (p95 333-549) |
| NPU | Qwen3.5 hybrid attention does not run (driver errors) |
Non-hybrid models (Gemma 4 E2B/E4B, Llama-3.2-class) might run on the NPU, but no measurement
exists; the iGPU path is proven. A second model on the RTX costs ~2-4 GB VRAM (fits in 24 GB with
Gemma at ~13 GB) but contends for compute. Arc estimates for E2B generation speed on discrete Arc
cards are model-based guesses (~30-65 tok/s) and say nothing about the 140T; measure.

### Filler and early speech
- LiveKit Agents: *preemptive generation* starts the LLM before end-of-turn is confirmed (TTS
  waits unless `preemptive_tts`); discarded when context changes, with extra token cost; docs say
  it "doesn't guarantee reduced latency"
  ([LiveKit docs](https://docs.livekit.io/agents/multimodality/audio.md)). 1.6.0 added
  asynchronous tools to keep the user informed during long tool calls; a forced "announce before
  the tool call" line is still a gap people solve with custom `llm_node` code
  ([forum](https://community.livekit.io/t/make-the-agent-announce-prior-to-calling-the-tool-functionality-not-available/1434)).
  Pipecat filler: nothing found.
- Grace already announces finished background jobs and glows the screen while tools run (git
  log); there is no data here on how users react to canned fillers.

## 3. Fit for Grace
- **Router in front of the 12B is the weak spot:** the router itself costs 130-300 ms, a large
  share of an 800 ms budget, and Grace's failures are in *tool selection*, which a 1-4B model does
  worse than the 12B. A router that wrongly sends a tool request to the small model is a new
  failure class. Savings only exist for turns that currently run through the 12B for no reason.
- **Cheaper than a model router:** (a) deterministic fast path in grace-core for exact commands
  ("pause", "next", "volume up", Danish variants, matched on the STT text or via harrier
  embeddings on CPU) that skips the LLM entirely; (b) the existing JevK5-on-iGPU decision model
  extended with a "trivial intent" option, already off the RTX with ~1.5% decode cost.
- **Cascade the other way:** let the 12B answer and use the small model only for decisions
  (that is what Grace does). Speculative "small answers first, escalate on low confidence" needs
  a confidence signal that tiny models do not give reliably (RouteLLM-style training needed).
- **Early speech:** a pre-synthesised pool of short English acknowledgements ("One moment.",
  "Let me check.") played the moment a tool call is detected (or when the first sentence is not
  ready after N ms) costs no LLM or GPU time and fits the Pocket TTS streaming path. Risks: the
  echo/false-trigger rule (it is Grace's own voice, already covered by ADR 0006 handling), wrong
  or repetitive fillers feeling robotic, and a filler delaying the real answer if it is not
  interruptible/cancellable. Preemptive generation on partial STT is higher gain but doubles LLM
  work and conflicts with the prompt-cache slot layout; do not start there.

## 4. Expected gain and costs
- Deterministic fast path: for matched commands, saves the full LLM time (hundreds of ms to
  ~1 s) with no VRAM; coverage unknown until measured on traces (share of turns that are
  commands).
- Model router to E2B/E4B: gain only on chit-chat turns, maybe 100-300 ms; adds a model (2-4 GB
  VRAM or iGPU RAM), routing errors, more maintenance. Likely small net.
- Filler: improves *perceived* latency on tool turns by the length of the tool call; zero
  measurable latency change, so it needs a perception check, not a stopwatch.
- Licence: Gemma terms, FunctionGemma and Qwen all allow local use.

## 5. How we would test
1. From existing traces/live logs, label turns (command, chit-chat, tool lookup, complex) and
   count the share and current latency per class. If commands + chit-chat < 25% of turns, stop.
2. Fast path: run the tool/command rules against `grace-tools-sim` scenarios and the Danish STT
   transcripts; metric: precision (no false fast-path on non-command utterances, target 0 on
   sim) and recall; latency from trace.
3. Small-model router: score E4B and JevK5 on the same labelled turns as a classifier
   ("needs 12B?"), then measure grace-chat-bench reply quality for the turns it would take;
   adopt only if quality loss is within noise and total TTFT p50 improves >= 150 ms net of router
   time.
4. Filler: simulated-user harness with canned fillers on/off over tool turns; measure time to first
   audio, interruptions, and echo false-triggers (media false-trigger test must stay 0); then one
   short live check.

## 6. Verdict
Router to a smaller LLM: **not for us** (savings small, new tool-selection errors, router cost
rivals the gain). Deterministic fast path for exact commands: **adopt-candidate** after a
trace count of how many turns qualify. Canned acknowledgements while tools run: **maybe**
(cheap to build, needs the perception test). FunctionGemma/E2B on iGPU/NPU: **maybe**, only as
an experiment if the fast path proves too narrow.
