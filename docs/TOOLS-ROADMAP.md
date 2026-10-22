# Grace v2 — tools roadmap

Written 2026-10-22 after live run 9, at Mikkel's request: every tool and refinement Grace should
get, compared with what Grace v1 had (108 tools in `packages/tools/src/tools/`, see
`docs/3-TOOLS.md` in v1). v2 today: web_search (fact / overview / live / news), read_page,
weather, calculate, media, open_app, open_url, find_files, list_folder, open_file. Priorities:
**P1** next, **P2** soon after, **P3** later. Risk: *read* (just runs), *visible* (runs, says what
it did), *changes* (needs a spoken yes first; not built yet).

## 1. The tools she has: refinements

| Tool | Refinement | Why / evidence | P |
|---|---|---|---|
| media | Play playlists, albums and "something like X"; queue a song; like a song; set volume to an exact % | Run 9: "afspil noget Gilly" | P1 |
| media | Control YouTube/Chrome playback, not only Spotify | Run 8: YouTube on the speakers | P2 |
| media | Spotify Web API (needs Mikkel to create a Spotify app for a client id) for search results, devices, queue | Today's version opens track links and reads the window title | P3 |
| open_file | Reuse the open Explorer window (done in run-9 fixes); also go "up" a folder, back, select a file | Run 9 | P1 |
| open_app | Switch to / close a program, list what is running (v1: list_processes) | "luk spotify", "skift til chrome" | P1 |
| open_url | Open in the current tab, close tabs, read the current tab (needs Chrome DevTools or an extension) | Run 9: "åbn artiklen", then "opsummer" | P2 |
| find_files | Search inside files (v1: search_content), recent files (v1: recent_files), let him add folders to the index (v1: index_folders) | "hvad arbejdede jeg på i går" | P1 |
| find_files | Read a document: PDF, Word, Markdown (v1: read_document) to summarise or answer from it | Natural follow-up after finding a file | P1 |
| calculate | Units and currencies (live rates), dates ("hvor mange dage til jul"), a rounding preference (2 decimals unless asked) | Run 9: "2 decimaler" | P2 |
| weather | Hourly ("regner det klokken 16?"), the week ahead, place from memory | Run 9 | P2 |
| web_search | Remember sources across turns and say them by name; a deeper multi-step research mode (v1: research, multi_hop_research, fact_check) | Run 7: "hvor læste du det" | P2 |
| web_search | Wikipedia directly (v1: wikipedia), definitions and translation (v1: define, translate) | Faster than general search for those | P3 |
| list_folder | Sort by date, filter by type ("hvilke PDF'er") | Run 9 counting | P3 |

## 2. New tools

### Computer control (P1: Mikkel asked for these in run 9)
- **clipboard_read / clipboard_write** (v1: clipboard) — *read* / *visible*.
- **type_text**: type or paste text where the cursor is (v1: inject_text, with a preview / confirm step) — *changes*: say it, then type on his yes.
- **notes**: save and list notes (v1: notes); write into Notesblok (run 9: "kan du skrive i min notepad").
- **active_context**: what window is in front, selected text (v1: get_active_context), so "opsummer det her" works.
- **screen**: screenshot + read the screen with Gemma 4's vision or OCR (v1: take_screenshot, analyze_screen, ocr_screen, read_screen_document).
- **system_status**: CPU, RAM, GPU, battery, disk (v1: system_status, disk_usage).
- **volume / brightness / lock / sleep** — *visible*.

### Time and organisation (P1)
- **reminders and timers** with natural language and Danish ("om tyve minutter", "på fredag kl 9"), spoken when due, surviving restarts (v1: reminders). Memory commitments already do "bring it up next time".
- **todos** (v1: todo) and a **focus timer** (v1: focus).
- **calendar** read and create, create only after a spoken yes (v1: calendar). A Google Calendar connector exists.
- **time / date** questions without the LLM guessing (v1: get_current_time).

### His world (P2)
- **itslearning / Docta**: deadlines, assignments and course news (the itslearning MCP exists).
- **mail**: read and summarise, draft replies, send only on a spoken yes (v1: gmail, draft_email).
- **news feeds**: chosen RSS sources (v1: rss, news_fetcher), a morning brief.
- **smart home**: Hue lights and Home Assistant (v1: hue, home_assistant), if he has them.

### Developer helpers (P2)
- **git status / commit message / CI status** (v1: git_repo, generate_commit_msg, ci_status).
- **run a command** from an allow-list (v1: run_command) — *changes*.
- **inject_to_terminal** (v1) — *changes*.
- **project overview** (v1: index_projects, list_workspaces).

### Later (P3)
- MCP client so connectors (Calendar, Gmail, itslearning) plug in without custom code (v1: mcp).
- Learning and wellness (v1: teach_me, eye_pause, habits, goals, mood, wellness), story mode, dice.
- Image generation (v1 wishlist).
- Self-extension (v1: create_tool) — only with tests and his approval.

## 3. The tool framework
- **A spoken "yes" for anything that changes data** (send, write, delete, create, type): Grace says
  exactly what she will do and waits for his yes. Nothing does this yet; it blocks type_text,
  mail, calendar-create and file editing.
- **Undo** for file moves and edits (v1: undo, verify_mutation).
- **Background tasks** with spoken progress and results at a pause (#21): long research, big file
  searches.
- **Tool manifest with hot reload** (#17): add a tool without restarting the engine.
- **Per-tool tests** in grace-tools-sim for every new tool (the rule since run 6), plus a live check.
- **Speed**: each tool says how long it usually takes, so Grace only says "Let me check" when it
  will take a while.
- **Tool results kept in the history** (done after run 7) and trimmed when long.
