# Building a tool group (for agents)

Read ADR 0021, `services/grace_tools/registry.py`, `tests/tools/conftest.py` and the built-in
tools in `services/tools_server.py` first.

## You own (and touch nothing else)
- `services/grace_tools/<group>.py` (more modules as `<group>_*.py` if needed)
- `services/grace_tools/<group>.requirements.txt`: pip packages you added to
  `bench/.venv-openvino` (install with that venv's python: `-m pip install ...`)
- `tests/tools/test_<group>.py` (+ fixtures under `fixtures/tools/recorded/<group>/`)
- `fixtures/tools/scenarios/<group>.yaml` (same format as `fixtures/tools/scenarios.yaml`)
- `docs/tools/<group>.md`: each tool, its risk, what you tested, what you could not test and why
Never edit tools_server.py, registry.py, conftest.py, the Rust engine, config/prompts, or
another group's files. If you need a change there, say so in your final report.

## Rules
- **Local only.** No cloud AI or paid APIs. The internet only for things like downloads or search.
- **Risk.** read = only looks. visible = changes something he sees and can undo at once.
  destructive = deletes, overwrites, sends, pushes, installs, kills, or changes a setting that is
  hard to undo; it needs a `summary(args)` such as "delete the 3 PDFs in Downloads". When unsure,
  choose destructive. Never build anything that spends money or types passwords or card details.
- **Honour `ctx.dry`**: in dry-run, report exactly what would happen and change nothing.
- **Results are for Gemma**: short English text. Give facts she will be asked next, like paths,
  counts and names. Results are spoken, so no tables or long dumps (about 1500 characters max).
  End with what failed and why when something did.
- **Descriptions** say when to use the tool, in English, starting with a capital letter.
- **Where files go**: new files go where he asked; if he didn't say, `~/Documents/Grace/`.
  Never overwrite without a destructive call.
- **Tests are deterministic.** No internet, no real user folders, no real windows, processes or
  settings. Put OS calls in small functions that tests replace with monkeypatch. Cover the
  normal case, Danish input where it applies, failures and dry-run.
  Run: `bench/.venv-openvino/Scripts/python.exe -m pytest tests/tools -q`
- **Sim scenarios**: invented Danish lines as STT delivers them (lowercase, no punctuation),
  5-15 per group, including multi-step tasks, and one destructive case where the tool must ask
  first. Grace reaches non-core tools through `find_tools` then `use_tool`, so `tools:` lists
  those (e.g. `[find_tools, use_tool]`), with `args` checking use_tool's inner name. Don't run
  the sim; the organiser does.
- **core=True** only for a tool needed in many everyday turns (at most 2 per group).
- **Code style**: match tools_server.py: short comments explaining why, plain functions.
- Commit on your branch with clear messages. Don't push or merge.
