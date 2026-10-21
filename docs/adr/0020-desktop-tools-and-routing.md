# 0020 — Desktop tools, a tools server, and routing requests with the checker

- **Status:** Accepted
- **Date:** 2026-10-21
- **Extends:** [0019](0019-web-lookup.md)

## Context
Live run 6 (the first with web search) failed where it mattered: Grace said "I don't know that
show" instead of searching, reacted to a search result as if Mikkel had said it ("That sounds
intense"), and found no Odense news, because the loop's checker asked "does this state the exact
answer?" of news pages. Opening programs, Chrome tabs and files, which Mikkel had asked for, was
missing. Gemma also can't multiply (17 x 23 -> 51, 510) and misreads Danish numbers ("treogtyve"
-> 30).

## Decision
- **One tools server** (`services/tools_server.py`, port 7813, replaces lookup_server.py) serves a
  manifest and runs every tool: `web_search` (types fact / overview / live / news), `weather`
  (Open-Meteo, free), `calculate` (reads Danish number words itself; arithmetic only),
  `open_app` (Start menu apps; an unknown name opens the website instead), `open_url` (new Chrome
  tab), `find_files` / `list_folder` / `open_file` (user folders, indexed in a child process;
  never opens programs or scripts). Nothing deletes, sends or buys. `--dry-run` for benches.
- **News** is read from the news sites' front pages (SearXNG's news engines don't answer);
  **overview** questions must match the exact title; **live** values stop after one round.
- **The engine's tool loop** (`tools::reply`) is shared by Grace and the benches: up to 3 calls
  per reply, a filler line before slow tools, tools run only after the reply is committed (a
  prepared reply that is thrown away never searches or opens anything), the server stops a call
  when the engine drops it. Tool results are framed as "what you found, not what Mikkel said".
- **Routing:** the checker's needs-a-tool decision (search / weather / calculate / open / files /
  none) is started as soon as the words are transcribed. "search" above 0.5 adds the lookup hint;
  "none" at 0.8 or more sends the request without tool definitions, so plain conversation looks
  as it did before tools. In Gemma's template the tool declarations follow the system text, so
  the memory block now comes first in the system message.
- **Prompt:** search instead of "I don't know"; act without asking "should I...?" first.

## Consequences
- grace-tools-sim (27 lines x 3, Mikkel's style rules, real search and file search, dry-run
  actions): 81/81 before routing; chat bench with tools 0.966 (0.956 before tools, 0.929 with
  tools offered on every turn). Details in docs/MODELS.md.
- Search engines rate-limit a burst of searches (benchmarks): SearXNG now spreads over more
  engines, suspends a blocked one for 5 minutes instead of an hour, and the server caches results
  for 10 minutes.
- Opening programs and files happens without a confirmation; anything that changes data will need
  a spoken yes (not built). Search results and file names reach Gemma only as quoted data.
