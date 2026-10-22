# ADR 0021 — Tool registry, risk classes and tool discovery

Date: 2026-10-22. Status: accepted.

## Context
Mikkel asked for about 60 more tools (docs/TOOLS-ROADMAP.md) and for Grace to choose and chain them
herself, asking only before destructive actions. One monolithic `services/tools_server.py` cannot
be built by several agents at once. Every tool definition in every request costs prompt tokens,
and changing the tool list per turn breaks llama-server's prompt cache (live run 8).

## Decision
- **Registry.** Tool groups live in `services/grace_tools/<group>.py` and register with
  `@tool(name, description, params, required, risk, slow, core, summary)`. The package loads every
  module at start. The ten built-in tools stay in tools_server.py.
- **Risk classes**, enforced by the server, never by the model: `read`, `visible`, `destructive`.
  A destructive call returns only a summary and a token. The engine runs it (`{"kind":"confirm"}`)
  only after Mikkel's own next utterance is a plain yes. Spending money and typing passwords or
  card details are not tools at all.
- **Discovery.** Built-in and `core=True` tools are offered on every turn; the rest are found with
  `find_tools(need)` and run with `use_tool(name, args)`. The list Gemma sees never changes, so
  the cache holds. `--all-tools` offers everything directly so the two can be measured.
- **Tests.** Every group has `tests/tools/test_<group>.py`, deterministic: no internet
  (blocked in conftest), only tmp_path files, OS actions behind a thin replaceable layer.
  It also has sim scenarios in `fixtures/tools/scenarios/<group>.yaml` for grace-tools-sim.

## Consequences
One extra round-trip (find, then use) for non-core tools; measured against `--all-tools` in the
tools sim before this is final. Groups can be added without touching the engine.
