# 0010 — Tools live behind a process boundary

- **Status:** Accepted (design); implementation starts in v0.3
- **Date:** 2026-10-10

## Context
New tools must go live without restarting Grace, and later Grace will write tools for herself.
In v1, self-written tools were compiled into the same process, edited Grace's own source, and
needed elaborate guards to stay safe.

## Decision
- A tool is a **manifest** (name, description, JSON schema, risk level) plus an executable that
  speaks a JSON-RPC protocol over stdio, MCP-compatible where practical. Any language.
- The harness watches the tools directory: adding, changing or removing a tool takes effect on
  the next turn.
- Core tools needed early (web search, open program, file search) may be built into the
  harness behind the same interface, and can move out later.
- Risk levels: `auto` (read-only), `confirm` (needs a fresh yes from the user in the
  conversation), `manual` (Grace explains, the user does it). The model can't grant itself
  confirmation.
- The model sees a small, relevant subset of tools per turn, not the whole catalogue.

## Consequences
- Process spawn cost per call; long-lived tool processes amortise it.
- A natural sandbox boundary for self-written tools (restricted working directory, no network
  unless the manifest declares it).
