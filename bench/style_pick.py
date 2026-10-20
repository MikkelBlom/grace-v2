"""Blind style pick: answers the short exchanges in fixtures/chat/pick/ in each candidate style
(fixtures/chat/styles/*.md) through grace-chat-bench, scores each style on the full scenario
set, and writes a blind pick page where Mikkel chooses the reply he likes best.

    python bench/style_pick.py [--seeds 2] [--skip-full] [--page-only]

Each style is the current config/prompts/system.md with only its personality paragraph replaced,
so every rule (quiet, interview, length) stays identical. Needs llama-server and a release build
of grace-chat-bench. Output: bench-results/chat/style-pick/pick.html (picks persist in the
browser; "Copy picks" puts them on the clipboard as JSON for the prompt).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "target" / "release" / "grace-chat-bench.exe"
STYLES = ROOT / "fixtures" / "chat" / "styles"
OUT = ROOT / "bench-results" / "chat" / "style-pick"
# The paragraph each style replaces: from "Personality (" up to this sentence.
PERSONALITY = re.compile(r"Personality \(.*?Vary your wording; don't reuse the same phrases\.", re.S)


def style_prompt(base: str, block: str) -> str:
    new, n = PERSONALITY.subn(lambda _: block.strip() + "\n\nVary your wording; don't reuse the same phrases.", base)
    if n != 1:
        sys.exit("system.md: personality paragraph not found (style_pick.py's PERSONALITY regex)")
    return new


def run_bench(label: str, prompt: Path, extra: list[str]) -> dict:
    subprocess.run([str(BENCH), "--label", label, "--system-prompt", str(prompt), *extra], cwd=ROOT, check=True)
    return json.loads((ROOT / "bench-results" / "chat" / label / "report.json").read_text(encoding="utf-8"))["summary"]


def last_exchanges(label: str) -> dict[str, dict]:
    """The final (answered) exchange of each pick scenario."""
    out: dict[str, dict] = {}
    for line in (ROOT / "bench-results" / "chat" / label / "transcripts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            out[e["scenario"]] = e
    return out


def page(items: list[dict], scores: dict[str, dict]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    score_rows = "".join(
        f"<tr><td>{html.escape(s)}</td><td>{v.get('score', '')}</td><td>{v.get('hard', '')}</td><td>{v.get('judge', '')}</td></tr>"
        for s, v in scores.items()
    )
    return PAGE.replace("__DATA__", data).replace("__SCORES__", score_rows)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grace Style Pick</title>
<style>
:root { --bg:#0f1115; --panel:#161a21; --line:#252b36; --text:#d7dce4; --muted:#7d8696; --accent:#e8b04a; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,sans-serif; }
main { max-width:860px; margin:0 auto; padding:20px 16px; }
h1 { font-size:17px; margin:0 0 4px; } p { color:var(--muted); margin:4px 0 10px; }
.s { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:12px 0; }
.ctx { color:var(--muted); font-size:13px; } .me { margin:4px 0 10px; } .me b { color:var(--accent); }
label.c { display:flex; gap:10px; align-items:flex-start; padding:8px; border-radius:6px; cursor:pointer; border:1px solid transparent; }
label.c:hover { border-color:var(--line); } label.c input { margin-top:4px; }
label.c.on { border-color:var(--accent); }
.k { font-family:monospace; color:var(--muted); width:16px; } .lbl { font-family:monospace; color:var(--accent); visibility:hidden; margin-left:auto; padding-left:10px; }
.rev .lbl { visibility:visible; } .rev #scores { display:table; }
textarea { width:100%; box-sizing:border-box; background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:6px; font:inherit; font-size:13px; padding:6px; margin-top:6px; }
button { font:inherit; padding:6px 12px; margin:0 8px 6px 0; }
.bar { position:sticky; top:0; background:var(--bg); padding:10px 0; z-index:1; border-bottom:1px solid var(--line); }
#scores { display:none; border-collapse:collapse; margin:10px 0; font-size:13px; } #scores td, #scores th { border:1px solid var(--line); padding:4px 8px; }
#out { position:fixed; right:16px; bottom:16px; max-width:min(440px, calc(100vw - 32px)); max-height:60vh; overflow:auto; margin:0; padding:12px 14px;
  background:var(--panel); border:1px solid var(--accent); border-radius:8px; white-space:pre-wrap; display:none; font-size:13px; }
</style></head><body><main>
<h1>Which Grace do you like?</h1>
<p>Each exchange shows what you said and four of Grace's replies, in random order. Pick the one you like best, or "none". A note is optional ("too stiff", "this but shorter"). Picks are saved in this browser. When done, press "Copy picks" and paste the result to Claude.</p>
<div class="bar"><button id="copy">Copy picks</button><button id="rev">Reveal styles</button><span id="count"></span></div>
<table id="scores"><tr><th>style</th><th>bench score</th><th>hard</th><th>judge</th></tr>__SCORES__</table>
<div id="list"></div><pre id="out"></pre>
<script>
const items = __DATA__;
const KEY = 'grace-style-pick';
const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; } };
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch {} };
const state = load();
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const list = document.getElementById('list');
for (const it of items) {
  const st = state[it.id] ||= {};
  const opts = shuffle(it.replies.slice());
  const s = document.createElement('div'); s.className = 's';
  s.innerHTML = (it.before ? `<div class="ctx">Grace before: ${esc(it.before)}</div>` : '') +
    `<div class="me"><b>You:</b> ${esc(it.user)}</div>` +
    opts.map((o, k) => `<label class="c"><input type="radio" name="${it.id}" value="${o.style}"><span class="k">${String.fromCharCode(65 + k)}</span><span>${esc(o.reply || '(silent)')}</span><span class="lbl">${o.style}</span></label>`).join('') +
    `<label class="c"><input type="radio" name="${it.id}" value="none"><span class="k">–</span><span>None of them</span></label>` +
    `<textarea rows="1" placeholder="Note (optional)"></textarea>`;
  list.appendChild(s);
  const radios = [...s.querySelectorAll('input')];
  const mark = () => radios.forEach(r => r.parentElement.classList.toggle('on', r.checked));
  radios.forEach(r => { if (r.value === st.pick) r.checked = true; r.onchange = () => { st.pick = r.value; save(); mark(); count(); }; });
  const note = s.querySelector('textarea'); note.value = st.note || '';
  note.oninput = () => { st.note = note.value; save(); };
  mark();
}
const count = () => { document.getElementById('count').textContent = `${items.filter(i => state[i.id]?.pick).length}/${items.length} picked`; };
count();
document.getElementById('rev').onclick = () => document.body.classList.toggle('rev');
document.getElementById('copy').onclick = async () => {
  const tally = {};
  const picks = items.filter(i => state[i.id]?.pick || state[i.id]?.note).map(i => {
    const st = state[i.id], r = i.replies.find(x => x.style === st.pick);
    if (st.pick) tally[st.pick] = (tally[st.pick] || 0) + 1;
    return { id: i.id, user: i.user, pick: st.pick || null, reply: r ? r.reply : null, note: st.note || '' };
  });
  const text = JSON.stringify({ tally, picks }, null, 1);
  let copied = false;
  try { await navigator.clipboard.writeText(text); copied = true; } catch {}
  const out = document.getElementById('out');
  out.textContent = (copied ? 'Copied to clipboard.\\n\\n' : 'Copy this:\\n\\n') + text;
  out.style.display = 'block';
};
</script></main></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2, help="seeds for the full scenario run per style")
    ap.add_argument("--skip-full", action="store_true", help="don't score styles on the full scenario set")
    ap.add_argument("--page-only", action="store_true", help="rebuild the page from existing runs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = (ROOT / "config" / "prompts" / "system.md").read_text(encoding="utf-8")
    # "current" is system.md unchanged: the control, shown blind like the others.
    styles = sorted(p.stem for p in STYLES.glob("*.md")) + ["current"]
    scores: dict[str, dict] = {}
    for s in styles:
        prompt = OUT / f"system-{s}.md"
        if not args.page_only:
            text = base if s == "current" else style_prompt(base, (STYLES / f"{s}.md").read_text(encoding="utf-8"))
            prompt.write_text(text, encoding="utf-8")
            run_bench(f"style-{s}-pick", prompt, ["--scenarios", "fixtures/chat/pick", "--seeds", "1", "--no-judge"])
            if not args.skip_full:
                scores[s] = run_bench(f"style-{s}", prompt, ["--seeds", str(args.seeds)])
        elif (ROOT / "bench-results" / "chat" / f"style-{s}" / "report.json").exists():
            scores[s] = json.loads((ROOT / "bench-results" / "chat" / f"style-{s}" / "report.json").read_text(encoding="utf-8"))["summary"]
    by_style = {s: last_exchanges(f"style-{s}-pick") for s in styles}
    items = []
    for sid, first in sorted(by_style[styles[0]].items()):
        items.append({
            "id": sid,
            "user": first["user"],
            "before": first["before"],
            "replies": [{"style": s, "reply": by_style[s][sid]["reply"]} for s in styles if sid in by_style[s]],
        })
    (OUT / "pick.html").write_text(page(items, scores), encoding="utf-8")
    print(f"{len(items)} exchanges x {len(styles)} styles -> {OUT / 'pick.html'}")
    for s, v in scores.items():
        print(f"{s:<10} score {v['score']}  hard {v['hard']}  judge {v['judge']}")


if __name__ == "__main__":
    main()
