# 0019 — Web lookup: Gemma's web_search tool, a JevK5 checker on the Arc, and a search loop

- **Status:** Accepted
- **Date:** 2026-10-21

## Context
Live run 5 showed Grace inventing film facts with confidence. Mikkel wants a proactive assistant
whose lookups keep searching until they find the answer instead of taking the first hit, with
free search only (no paid APIs). Measurements (docs/MODELS.md, 2026-10-21):
- Gemma 4 12B with a `web_search` tool searched on 14/17 lookup requests and 0/33 others; the
  misses are exactly her confident-wrong cases (Sean Parker's actor, Runner Runner).
- JevK5 4B, a typed-decision (Jev-style) model, ties or beats Gemma on Grace's decisions
  (147-148/154), and its errors barely overlap with Gemma's. On the RTX it slows Gemma's decode
  50 -> 33 tok/s; on the Arc iGPU (OpenVINO INT8) 50 -> 48.5, at ~190-240 ms per decision. The
  NPU can't run it yet (Qwen3.5's hybrid layers crash the driver).
- With the checker's "needs a lookup?" hint on the user message, Gemma searched 17/17 and 0/33.
- The loop (search, judge snippets, read pages, refine the query) found the answer 17/18 times;
  the first search hit had it 9/18. Median 2.4 s, worst ~12 s.

## Decision
- `services/lookup_server.py` (port 7813) runs JevK5 4B INT8 on the Arc iGPU and the lookup loop.
  Search is SearXNG in Docker (`docker/searxng`), then DuckDuckGo's HTML page, then Wikipedia.
  The loop stops when the checker is >= 0.85 sure a snippet or passage answers the question, or
  after 3 rounds / 12 s; Gemma writes the refined queries.
- Every reply request offers Gemma one tool, `web_search(question, query)`. The checker's
  needs-lookup probability runs next to memory recall (400 ms cap) and adds a hint line to the
  user message above 0.5. At most one search per reply; Grace says a short filler line first.
- The tool result is evidence text without URLs; Gemma answers from it and says when nothing
  was confirmed. Only the spoken words enter the history.

## Consequences
- Replies that search take ~2-5 s more (the filler covers the wait); hard questions up to ~13 s.
  Every turn pays up to ~150 ms for the hint when the checker is slower than recall.
- Three more processes at startup (Docker/SearXNG, the lookup server); Grace runs without them.
- Search queries leave the machine (to SearXNG's engines); inference stays local.
- Watch: plot/"what is it about" questions often don't reach the 0.85 bar and use the full
  budget; search results are untrusted data and go to Gemma only as quoted evidence.
