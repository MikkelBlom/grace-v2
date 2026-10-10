# Sessions

## 2026-10-10 — Claude Code (Opus 5.5): status of v1, v2 foundation (v0.0)
**Summary:** Audited Grace v1 (Documents/Claude/Projects/AI automation/grace): turns 17–40 s,
non-streaming JSON tool loop, ~100 tools, untested features; lessons in docs/LESSONS-FROM-V1.md.
Agreed with Mikkel on a ground-up v2 in Rust in this folder (old spec HTML + empty .slnx moved to
the Recycle Bin). Researched the Oct 2026 model landscape (STT, TTS, LLM, embeddings, AEC,
turn detection). Built the v0.0 foundation: workspace + CI, grace-trace, grace-debug,
grace-core turn machine, grace-bench + Python runners, FLEURS fixtures. Ran the first STT and
Danish TTS bake-offs (docs/MODELS.md). Wrote ADRs 0001–0010, roadmap, README, created the
public repo MikkelBlom/grace-v2 with issues/milestones.

**Issues encountered:** CoRal v3 is gated (needs Mikkel's HF access request). Chatterbox deps
pulled a CPU torch into its venv (fixed by uninstalling it). Windows cp1252 pipes broke æøå in the
runner protocol (fixed). The `gh` token lacks the `project` scope, so the board needs
`gh auth refresh -s project`.

**Next steps:** CoRal + own-voice STT fixtures; blind TTS listening test; then v0.1 audio engine
starting with the AEC bake-off.
