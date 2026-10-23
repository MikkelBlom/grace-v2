# Caching and early action (R1: caching-and-early-action)

Desk research, 2026-10-23. Many sources are vendor blogs; PDFs of the two papers were unreadable, so
only abstracts were used.

## 1. What it is
Ways to skip or overlap work on the voice path: (a) cache replies or tool results, (b) run read-only
tools early on partial transcripts, (c) start the LLM before the turn is confirmed (preemptive
generation), (d) full-duplex speech models that avoid the cascade. Grace's turn detector already
prepares replies before the commit, so the question is what else fits on top.

## 2. Evidence
**Semantic caching of replies.** An embedding lookup returns a stored answer if the new prompt is close
enough. Vendor/blog claims: hits cost 5-20 ms instead of 0.8-3 s; usual thresholds 0.85-0.95; at 0.80 opposite
intents ("cancel" vs "keep") collide; advice is start strict, track hit rate and false-hit rate together, never
cache live or personalised answers. Nothing voice-specific or Danish/STT-noise specific was found.
https://www.truefoundry.com/glossary/what-is-semantic-caching-and-how-does-it-work ,
https://futureagi.com/glossary/semantic-cache/ , https://futureagi.com/blog/what-is-semantic-caching-llms-2026/

**Tool-call memoisation.** No good source; plain engineering: key = tool + normalised args, short TTL, only
for pure read-only tools (weather, time zones, calendar reads with an invalidation event).

**Early execution of read-only tools.** Stream recommends speculating only on read-only tools (weather, balances,
calendar), never on state-changing ones, because a wrong read costs compute while a wrong write may be
irreversible; fire during VAD/transcription when intent is highly probable. FutureAGI claims 200-400 ms at p95
(not benchmarked). Corti's beta guide says keep the speculative call stateless and replace it with the real
result once the turn is final (experimental). https://getstream.io/blog/speculative-tool-calling-voice/ ,
https://futureagi.com/blog/how-to-optimize-livekit-latency-2026/ , https://docs.corti.ai/stt/guides/agent-conversational .
Research: "Speculative Interaction Agents" (arXiv 2605.13360): 1.3-1.7x speedups on cloud real-time APIs with minor
accuracy loss, 1.6-2.2x on 3B edge models after training (abstract only; note the training method needs a
clock-based tuning step). https://arxiv.org/abs/2605.13360

**Preemptive generation.** LiveKit starts only the LLM on the preflight transcript before end-of-turn is
confirmed, TTS waits (unless `preemptive_tts`); if chat context or tools change, the draft is discarded and
regenerated; skipped beyond 10 s of speech; max 3 retries; costs extra tokens. Community thread: it did not
fire for some STT models. https://docs.livekit.io/agents/build/speech/ ,
https://community.livekit.io/t/livekit-preemptive-generation-not-triggering-with-current-stt-model-missing-preflight-transcript-events/1582 .
Pipecat: Smart Turn v3 (local ONNX turn detector) with short VAD stop (0.2 s default); v1.3.0 notes mention
eager end-of-turn events (turn.eager_end / turn.resume), whether they start the LLM early was not confirmed.
https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview ,
https://newreleases.io/project/pypi/pipecat-ai/release/1.3.0

**Full-duplex models.** Moshi: ~160 ms theoretical, ~200 ms measured on an L4; a single speech model, no tools.
https://huggingface.co/docs/transformers/v5.13.1/model_doc/moshi . NemotronLabs VoiceChat 11B: ~448 ms
turn latency (vendor), tool calling on a separate channel with "on-hold" filler speech while an API call resolves; but
BFCL-v3 average 56.1%, right tool 82.5% yet right arguments 44.2% on Full-Duplex-Bench 3.0, max about 5 tools per
session, no reliable parallel calls, research-leaning licence (OpenMDW). So the latency win comes with a large
tool-quality loss versus a 12B text LLM with 130 tools. https://aiweekly.co/alerts/nvidia-opens-nemotronlabs-voicechat-11b-with-tool-calling ,
https://neurips.cc/virtual/2026/166988 . No independent Pipecat vs Moshi comparison found.

## 3. Fit for Grace
- **Reply cache: no.** English replies depend on persona, history, time and tools; STT of Danish is noisy, so
  near-duplicate detection is unreliable, and a false hit says the wrong thing aloud. Only a closed set of
  canned acks (see below) is safe, and those need no semantic matching.
- **Tool memoisation: yes, small.** Tool layer, no model change. Key on tool + canonical args, TTL per tool
  (seconds to minutes), never for destructive or time-sensitive-by-second tools. Value is on repeat asks and
  on the speculative path (below), where the speculative result is reused if the real call matches.
- **Speculative read-only tools: fits, with guard rails.** The existing Suggester (129 ms, 54/56) can run on
  the partial transcript the turn detector already has; if its top pick is a tool whose `risk` is read-only and its
  args can be derived without the model, warm it (or only warm network connections/caches). Arguments
  usually come from Gemma, so the realistic win is "start the fetch for tools with no or obvious args" (time,
  weather for default place, calendar today, unread mail count), and keep the result keyed for reuse. Never
  speculate on tools with side effects, `risk != read`, or anything needing a confirmation; the "model can't
  approve itself" rule stays untouched because nothing speculative is visible to the user. Discard silently if
  the final transcript changes the pick.
- **Preemptive generation: already the architecture.** Grace's turn detector prepares replies before the
  commit. What to add from LiveKit's design: discard-and-regenerate when the final transcript differs
  (compare text, not just time), skip for long utterances, cap retries, and make the llama.cpp request
  cancellable. Watch the prompt cache: cancelling mid-prefill and SWA checkpoints (R0 item 6) can cost the
  1.5 s misses, and a speculative run on `-np 2` can occupy the second slot the Suggester needs; measure
  slot contention.
- **Filler / on-hold line.** The transferable part of VoiceChat: for tool turns, speak a short fixed
  acknowledgement (pre-synthesised, English, a handful of variants chosen by tool group) while the tool
  runs. It is not caching; it hides latency at zero model cost and cannot be wrong if it says nothing
  about the result. Must stay interruptible and must not trigger her own echo path (ADR 0006).
- **Full-duplex swap: no.** Danish in, 130 tools, local Gemma persona and the 5-tool/44% argument accuracy of
  VoiceChat rule it out; PersonaPlex is already parked in FUTURE_IDEAS.

## 4. Expected gain and costs
- Tool memoisation: saves the tool round trip only on repeats; gain small and workload-dependent.
- Speculative read-only prefetch: bounded by tool latency (typically 100-500 ms for web/API tools, near
  zero for local ones). Vendor claims 200-400 ms at p95; for us only on turns that use such a tool.
- Filler line: cuts perceived (not actual) latency on tool turns, probably the biggest user-visible win here.
- Preemptive generation tuning: potentially the largest on all turns, but it is mostly already done; the gain
  now is in correctness and cache hygiene, not new speed.
- Risks: wasted GPU (power, slot contention), stale or wrong speculative results, cache complexity, extra traces
  needed (add `stage = "spec.hit"/"spec.miss"` spans, contract with grace-debug).

## 5. How we would test
- Replay the tools-sim scenario transcripts with injected partial transcripts (every N ms of the
  streamed STT output); log how often the Suggester's read-only pick at the partial equals the final pick and
  whether args equal. Adopt prefetch if the pick agreement is >= 95% on read-only tools and zero
  state-changing tools ever fire speculatively.
- A/B latency traces (TTFT, speech-end to first audio) with prefetch and filler on/off, plus slot-contention
  check under `-np 2`; also the media false-trigger test must stay at 0 with the filler line.
- Semantic cache only as a throwaway offline experiment: embed 1000 synthetic paraphrases with STT-style noise
  and measure false-hit rate at thresholds 0.85-0.95; unless it is zero for the idempotent subset, drop it.

## 6. Verdict
- Semantic reply cache: **not for us** (false hits are worse than the saved ms; Danish STT noise).
- Tool memoisation + read-only speculative prefetch: **adopt-candidate** (cheap, tool layer, rules-compatible).
- Filler/on-hold line during tool calls: **adopt-candidate** (best perceived-latency win).
- Preemptive generation hardening (discard on changed final, cancel safety, slot/cache checks): **adopt-candidate**
  as an audit of what exists.
- Full-duplex models (Moshi, VoiceChat): **not for us**.
