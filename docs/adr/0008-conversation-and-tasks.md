# 0008 — Conversation and tasks are separate actors

- **Status:** Accepted (design); implementation starts in v0.3
- **Date:** 2026-10-10

## Context
Tool work (web search, file search) can take seconds. Mikkel wants to keep talking meanwhile
and hear what Grace is doing, not sit through silence. In v1, missions and the conversation
shared mutable state in one LLM class, so interruptions and background work collided.

## Decision
- The **conversation actor** owns the floor. It always answers immediately, even if only with
  "I'll look that up".
- A **task actor** is spawned per piece of longer work, with its own context, cancellation token
  and budget. It never speaks directly; it emits progress events.
- The conversation actor decides whether to narrate progress ("searching three sources now"),
  and delivers the result when the task finishes, at a natural pause, not mid-sentence.
- `llama-server` runs with parallel slots so the conversation and tasks share one loaded model.

## Consequences
- No shared mutable state between actors; they communicate by messages only.
- "What are you working on?" and "stop that" are answered from the task registry, not by asking
  the model.
- Needs rules for delivering results that arrive while the user is speaking (queue until pause).
