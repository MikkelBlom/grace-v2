# Research brief: making Grace's small model smarter and faster

Started 2026-10-23 at Mikkel's request. Desk research only (R0 sweep, R1 topics); R2 ranks;
experiments (R3) only after he picks.

## Grace's setup (what every finding must be checked against)
- Laptop: RTX 5090 Laptop GPU (Blackwell, 24 GB VRAM), 64 GB RAM, Intel Core Ultra 9 with NPU
  and Arc 140T iGPU. Windows 11. It is also his work/school/personal machine.
- Main model: Gemma 4 12B instruct, Google's QAT Q4_0 GGUF, in llama.cpp `llama-server`
  (build b11539, CUDA), `-c 32768 -np 2` (16k per slot), about 13 GB VRAM; plus a vision
  projector (mmproj). About 50 tok/s decode.
- Voice path: Danish speech in (edda STT), English replies out (Pocket TTS), streaming per
  sentence. Target: under 800 ms from end of speech to first audio. Prompt cache matters: the
  system prompt + tool list is a fixed prefix (live run 8: cache misses cost up to 1.5 s).
- Tools: about 130 tools via a registry; Gemma picks from a one-line catalog (a second request to
  the same llama-server) and calls tools via OpenAI-style function calling, up to 8 per reply.
  Known weak spots (tools sim 57/162 with stale scenarios): loops of repeated calls with bad
  arguments, answering without a tool when one fits, picking a near-miss tool.
- Side models: JevK5 4B (Qwen3.5-based) INT8 on the Arc iGPU via OpenVINO for yes/no decisions;
  harrier-270m embeddings on CPU. The NPU could not run Qwen3.5 hybrid attention.
- Benches we measure with: grace-tools-sim (tool use), grace-chat-bench (reply quality),
  grace-style-bench, latency traces (TTFT, decode).
- Rules: local inference only (no cloud AI at runtime); no paid services; measure, don't guess.
  Training data must not come from Claude or Gemini output (their terms forbid training models on
  it); OK: programmatic generation, self-distillation with bench scoring, open-weight teachers
  whose licence allows it.

## What each topic file must contain (docs/research/small-models/<topic>.md)
1. What it is, in a few plain sentences.
2. Evidence: numbers from papers, benchmarks, release notes, issues; link every source. Say
   when a number is a vendor claim or from one person's post.
3. Fit for Grace: does it work on Windows + this GPU + llama.cpp (or what would have to change),
   with Gemma 4 12B, with tool calling and the prompt cache, Danish in / English out.
4. Expected gain (speed or quality, with a range) and costs/risks (VRAM, effort, maintenance,
   quality loss, licence).
5. How we would test it with our benches: the exact experiment and what result would make us
   adopt it.
6. Verdict: adopt-candidate / maybe / not for us, in one line, and why.
Be concrete and current (2025-2026 sources first). Short is fine; no padding. No time estimates.
