//! Blind listening page for TTS bake-offs: every sentence is played by every voice, in a
//! shuffled order, with the labels hidden until you reveal them. Ratings stay in the page
//! (copy them out with the "export" button) so nothing is uploaded anywhere.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde_json::{Value, json};

pub fn build(dir: &Path) -> anyhow::Result<PathBuf> {
    let mut voices = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path().join("report.json");
        if let Ok(text) = fs::read_to_string(&path) {
            let report: Value = serde_json::from_str(&text)?;
            voices.push(report);
        }
    }
    anyhow::ensure!(!voices.is_empty(), "no TTS reports under {}", dir.display());

    let data: Vec<Value> = voices
        .iter()
        .map(|r| {
            json!({
                "label": r["summary"]["label"],
                "ttfa_p50_ms": r["summary"]["ttfa_p50_ms"],
                "items": r["items"].as_array().cloned().unwrap_or_default()
                    .into_iter()
                    .map(|i| json!({ "id": i["id"], "text": i["text"], "audio": i["audio"] }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let page = TEMPLATE.replace("__DATA__", &serde_json::to_string(&data)?);
    let path = dir.join("listen.html");
    fs::write(&path, page)?;
    Ok(path)
}

const TEMPLATE: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grace TTS Listening</title>
<style>
:root { --bg:#0f1115; --panel:#161a21; --line:#252b36; --text:#d7dce4; --muted:#7d8696; --accent:#e8b04a; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,sans-serif; }
main { max-width:900px; margin:0 auto; padding:20px 16px; }
h1 { font-size:16px; } p { color:var(--muted); }
.s { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:12px 0; }
.t { margin-bottom:8px; } .c { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin:6px 0; }
.c b { width:22px; font-family:monospace; } .lbl { color:var(--accent); font-family:monospace; visibility:hidden; }
.rev .lbl { visibility:visible; } audio { height:32px; max-width:100%; } select { font:inherit; }
button { font:inherit; padding:6px 12px; margin-right:8px; }
</style></head><body><main>
<h1>Blind TTS comparison</h1>
<p>Each sentence is spoken by every voice in random order. Rate 1–5, then reveal. Ratings stay in this page.</p>
<button id="rev">Reveal labels</button><button id="exp">Export ratings</button>
<div id="list"></div><pre id="out"></pre>
<script>
const voices = __DATA__;
const ids = [...new Set(voices.flatMap(v => v.items.map(i => i.id)))];
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const list = document.getElementById('list');
for (const id of ids) {
  const takes = shuffle(voices.map(v => ({ label: v.label, item: v.items.find(i => i.id === id) })).filter(t => t.item));
  const s = document.createElement('div'); s.className = 's';
  s.innerHTML = `<div class="t">${takes[0].item.text.replace(/</g, '&lt;')}</div>` + takes.map((t, k) =>
    `<div class="c"><b>${String.fromCharCode(65 + k)}</b><audio controls preload="none" src="${t.item.audio}"></audio>
     <select data-id="${id}" data-label="${t.label}"><option value="">rate</option>${[1,2,3,4,5].map(n => `<option>${n}</option>`).join('')}</select>
     <span class="lbl">${t.label}</span></div>`).join('');
  list.appendChild(s);
}
document.getElementById('rev').onclick = () => document.body.classList.toggle('rev');
document.getElementById('exp').onclick = () => {
  const scores = {};
  document.querySelectorAll('select').forEach(s => { if (s.value) (scores[s.dataset.label] ||= []).push(+s.value); });
  const rows = Object.entries(scores).map(([l, v]) => `${l}: mean ${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(2)} over ${v.length}`);
  document.getElementById('out').textContent = rows.join('\n') || 'No ratings yet.';
};
</script></main></body></html>
"#;
