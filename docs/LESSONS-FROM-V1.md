# Lessons from Grace v1

Grace v1 (TypeScript, May–October 2026, 146 commits, ~100 tools) lives in a separate
repository and is kept as read-only reference. These are the lessons that shaped v2.

## Why v1 never felt usable
1. **Turns took 17–40 s.** The model filled a full `{thought, tool, args, speak, done}` JSON
   object before anything could be spoken, one tool per LLM call, with a 128K context and ~100
   tool descriptions in every prompt. → Stream, speak per sentence, small tool subset per turn.
2. **A hand-rolled tool protocol** instead of native tool calling: garbled tool names in
   batches, the same schema mistake repeated 4–5 times. → Use the model's native tool calling;
   give the model a working example, not just a schema.
3. **Breadth before depth.** Missions, self-written tools, Hue, Gmail and MCP all existed before
   one plain conversation felt good. → Milestones gated on a live test.
4. **Silence-only endpointing** cut Mikkel off or waited too long. → Turn-end model ([0005](adr/0005-turn-taking.md)).
5. **Shared mutable state** between missions and the conversation made interruptions collide.
   → Separate actors, per-turn cancellation ([0008](adr/0008-conversation-and-tasks.md)).
6. **2–3 minute launches** meant features went untested. → Process split, hot reload ([0002](adr/0002-process-split-and-hot-reload.md)).
7. **No per-stage timing**, so nobody knew where the seconds went. → Trace everything ([0003](adr/0003-observability-first.md)).

## What worked and is worth carrying over (as ideas, not code)
- Post-ASR correction dictionary for names Whisper reliably mishears ("klod" → Claude), and
  learning new corrections by voice. Soft hotword biasing pulls near-homophones, so pair it with
  the deterministic dictionary.
- Confidence gate on short safety-critical words (ja/nej before deleting something): re-ask
  instead of guessing.
- "Listen mode": hold the floor across pauses until an explicit end phrase.
- Safety rules: every chained shell segment must be read-only to auto-run; undo log entries
  re-validated before use; drafting an email never sends it; the model cannot set its own
  confirmation flag.
- Running STT on the Intel Arc iGPU (OpenVINO) kept the RTX free for the LLM.
- Gemma 4 26B and E4B could not co-reside with a second Gemma on 24 GB. Plan VRAM explicitly.
- Self-honesty prompts: "you are the worker, do it yourself", verify the artifact exists before
  claiming done, never state a value before the tool has returned.
