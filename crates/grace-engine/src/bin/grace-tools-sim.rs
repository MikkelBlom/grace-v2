//! `grace-tools-sim`: Grace's tool use end to end, without the microphone (ADR 0020).
//!
//!   grace-tools-sim --label v1 [--only s01] [--seeds 2] [--tools 127.0.0.1:7814]
//!
//! Each scenario in `fixtures/tools/scenarios.yaml` is a short conversation of Danish lines as
//! STT delivers them. Every line goes through the engine's own reply path: `system_message`,
//! the checker's needs-lookup hint (`Tools::hint`) and `tools::reply`, against the real
//! llama-server and a tools server. Run the tools server with `--dry-run` on its own port so
//! open_app / open_url / open_file report what they would do instead of doing it; web search
//! and file search run for real.
//!
//! Checked per turn: which tools were called (in order), their arguments, and the spoken reply
//! (regexes). Results: `bench-results/tools/<label>/` (transcript.md, rows.jsonl, report.json).

use std::{path::PathBuf, time::Instant};

use anyhow::Context;
use clap::Parser;
use grace::{conductor::system_message, memory, style, tools};
use grace_llm::client::{ChatRequest, LlmClient};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::Span;

const LLM_URL: &str = "http://127.0.0.1:8080";
/// Danish function words: two or more in a reply means she answered in Danish (ADR 0011). As
/// grace-chat-bench, plus words that showed up in tools sim v7 ("Her navn er Sofie.").
const DANISH_WORDS: &[&str] = &[
    "jeg", "ikke", "det", "og", "har", "du", "til", "af", "kan", "vil", "skal", "hvad", "så", "på",
    "er", "med", "mig", "dig", "der", "være", "også", "bare", "lige", "nej", "jo", "navn",
    "hedder", "hun", "han", "hendes", "hans", "din", "dit", "dine",
];

fn english(reply: &str) -> bool {
    let lower = reply.to_lowercase();
    let danish: std::collections::BTreeSet<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| DANISH_WORDS.contains(w))
        .collect();
    danish.len() < 2
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    label: String,
    #[arg(long, default_value = "fixtures/tools/scenarios.yaml")]
    scenarios: PathBuf,
    #[arg(long, default_value = "config/prompts/system.md")]
    system_prompt: PathBuf,
    #[arg(long, default_value = "127.0.0.1:7814")]
    tools: String,
    #[arg(long)]
    only: Option<String>,
    #[arg(long, default_value_t = 2)]
    seeds: u64,
    /// Time between the transcript and the reply start, when the engine has already started the
    /// needs-lookup check (`Tools::prefetch`). Live, the commit follows the transcript by
    /// ~0.3-1 s; 0 measures the check with no head start.
    #[arg(long, default_value_t = 300)]
    transcript_lead_ms: u64,
    /// Use the style rules from this memory database (read from a copy), as Grace would.
    #[arg(long)]
    rules_from: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    #[serde(default)]
    memory: String,
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    say: String,
    /// Tools that must be called, in this order (others may come between). `[]`: none at all.
    tools: Option<Vec<String>>,
    /// Regex per tool name over that call's JSON arguments.
    #[serde(default)]
    args: std::collections::BTreeMap<String, String>,
    /// Regex the spoken reply must match / must not match.
    answer: Option<String>,
    not_answer: Option<String>,
}

#[derive(Default)]
struct SimSink {
    text: String,
    first_text: Option<f64>,
    calls: Vec<(String, Value, Value)>,
    t0: Option<Instant>,
}

impl tools::Sink for SimSink {
    fn text(&mut self, text: &str) {
        self.first_text
            .get_or_insert_with(|| self.t0.map_or(0.0, |t| t.elapsed().as_secs_f64()));
        self.text.push_str(text);
    }
    fn filler(&mut self, line: &str) {
        self.text(line);
        self.text.push(' ');
    }
    fn tool_done(&mut self, name: &str, args: &Value, out: &Value) {
        self.calls.push((name.into(), args.clone(), out.clone()));
    }
    async fn cancelled(&mut self) {
        std::future::pending::<()>().await;
    }
    fn is_cancelled(&self) -> bool {
        false
    }
    async fn ready_to_act(&mut self) -> bool {
        true
    }
    fn silent(&self) -> bool {
        self.text.trim().is_empty()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let scenarios: Vec<Scenario> = serde_yaml::from_str(
        &std::fs::read_to_string(&args.scenarios).context("reading scenarios")?,
    )?;
    let llm = LlmClient::new(LLM_URL);
    let tools = std::sync::Arc::new(tools::Tools::connect(&args.tools).await?);
    let now = chrono::DateTime::parse_from_rfc3339("2026-10-21T14:10:00+02:00")?
        .with_timezone(&chrono::Local);
    let rules = match &args.rules_from {
        Some(db) => {
            let copy = std::env::temp_dir().join("grace-tools-sim-rules.db");
            std::fs::copy(db, &copy).context("copying the memory database")?;
            let mem = grace_memory::Memory::open(&copy, memory::EMBEDDER, memory::DIM)?;
            mem.style_rules()?
        }
        None => Vec::new(),
    };
    let out_dir = PathBuf::from("bench-results/tools").join(&args.label);
    std::fs::create_dir_all(&out_dir)?;
    let mut md = String::new();
    let mut rows = Vec::new();
    let (mut turns_n, mut tool_ok, mut args_ok, mut answer_ok, mut all_ok) = (0, 0, 0, 0, 0);
    let mut english_ok = 0;
    for sc in &scenarios {
        if args.only.as_deref().is_some_and(|o| !sc.id.starts_with(o)) {
            continue;
        }
        for seed in 1..=args.seeds {
            md.push_str(&format!("\n## {} (seed {seed})\n", sc.id));
            let block = format!("{}{}", sc.memory, style::render_rules(&rules));
            let system = system_message(&args.system_prompt, now, &block);
            let mut history: Vec<Value> = vec![json!({ "role": "system", "content": system })];
            for (ti, turn) in sc.turns.iter().enumerate() {
                let span = Span::none();
                if args.transcript_lead_ms > 0 {
                    tools.prefetch(&turn.say);
                    tokio::time::sleep(std::time::Duration::from_millis(args.transcript_lead_ms))
                        .await;
                }
                let t0 = Instant::now();
                let hint = tools.hint(&turn.say, &span).await;
                let mut content = style::with_reminder(&turn.say, &rules);
                if hint.is_some_and(|r| r.hint()) {
                    content = format!("{content}\n\n{}", tools::HINT);
                }
                let mut messages = history.clone();
                messages.push(json!({ "role": "user", "content": content }));
                let mut req = ChatRequest::new(messages);
                req.max_tokens = 300;
                req.extra.insert("seed".into(), json!(seed));
                let mut sink = SimSink {
                    t0: Some(t0),
                    ..Default::default()
                };
                tools::reply(
                    &llm,
                    hint.is_none_or(|r| r.offer_tools())
                        .then_some(tools.as_ref()),
                    req,
                    seed,
                    &turn.say,
                    &span,
                    &mut sink,
                )
                .await;
                let total = t0.elapsed().as_secs_f64();
                let reply = sink.text.trim().to_owned();
                history.push(json!({ "role": "user", "content": turn.say }));
                history.push(json!({ "role": "assistant", "content": reply }));

                let called: Vec<&str> = sink.calls.iter().map(|c| c.0.as_str()).collect();
                let t_ok = match &turn.tools {
                    None => true,
                    Some(want) if want.is_empty() => called.is_empty(),
                    Some(want) => {
                        let mut it = called.iter();
                        want.iter().all(|w| it.any(|c| c == w))
                    }
                };
                let a_ok = turn.args.iter().all(|(tool, re)| {
                    let re = Regex::new(re).expect("args regex");
                    sink.calls
                        .iter()
                        .any(|c| c.0 == *tool && re.is_match(&c.1.to_string()))
                });
                let r_ok = turn
                    .answer
                    .as_ref()
                    .is_none_or(|re| Regex::new(re).expect("answer regex").is_match(&reply))
                    && turn.not_answer.as_ref().is_none_or(|re| {
                        !Regex::new(re).expect("not_answer regex").is_match(&reply)
                    });
                let e_ok = english(&reply);
                turns_n += 1;
                english_ok += e_ok as usize;
                tool_ok += t_ok as usize;
                args_ok += a_ok as usize;
                answer_ok += r_ok as usize;
                all_ok += (t_ok && a_ok && r_ok && e_ok) as usize;
                let mark = if t_ok && a_ok && r_ok && e_ok {
                    "ok "
                } else {
                    "BAD"
                };
                let calls_txt: Vec<String> = sink
                    .calls
                    .iter()
                    .map(|(n, a, o)| {
                        format!(
                            "{n}({a}) -> {} {}",
                            o["status"]
                                .as_str()
                                .unwrap_or(if o["ok"].as_bool() == Some(true) {
                                    "ok"
                                } else {
                                    "fail"
                                }),
                            o["ms"]
                        )
                    })
                    .collect();
                println!(
                    "{mark} {} s{seed} t{} search={:.2} none={:.2} {:.1}s | {} | tools={called:?}{}{}{}",
                    sc.id,
                    ti + 1,
                    hint.map_or(-1.0, |r| r.search),
                    hint.map_or(-1.0, |r| r.none),
                    total,
                    turn.say,
                    if t_ok { "" } else { " [TOOLS]" },
                    if a_ok { "" } else { " [ARGS]" },
                    if r_ok { "" } else { " [ANSWER]" },
                );
                if !e_ok {
                    println!("      [DANISH]");
                }
                println!("      grace: {reply}");
                md.push_str(&format!(
                    "- **{mark}** Mikkel: {}  (search {:.2}, {:.1} s, first words {:.1} s)\n  - tools: {}\n  - Grace: {reply}\n",
                    turn.say,
                    hint.map_or(-1.0, |r| r.search),
                    total,
                    sink.first_text.unwrap_or(total),
                    if calls_txt.is_empty() { "none".into() } else { calls_txt.join("; ") }
                ));
                rows.push(json!({
                    "scenario": sc.id, "seed": seed, "turn": ti + 1, "say": turn.say,
                    "route": hint.map(|r| json!({ "search": r.search, "none": r.none })),
                    "tools": called, "calls": sink.calls.iter().map(|c| json!({"name": c.0, "args": c.1, "status": c.2["status"], "ok": c.2["ok"], "ms": c.2["ms"], "result": c.2["result"]})).collect::<Vec<_>>(),
                    "reply": reply, "total_s": total, "first_text_s": sink.first_text,
                    "tools_ok": t_ok, "args_ok": a_ok, "answer_ok": r_ok, "english_ok": e_ok,
                }));
            }
        }
    }
    let report = json!({
        "label": args.label, "turns": turns_n,
        "all_ok": format!("{all_ok}/{turns_n}"), "tools_ok": format!("{tool_ok}/{turns_n}"),
        "args_ok": format!("{args_ok}/{turns_n}"), "answer_ok": format!("{answer_ok}/{turns_n}"),
        "english_ok": format!("{english_ok}/{turns_n}"),
    });
    println!("\n{}", serde_json::to_string_pretty(&report)?);
    std::fs::write(out_dir.join("transcript.md"), md)?;
    std::fs::write(
        out_dir.join("rows.jsonl"),
        rows.iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )?;
    std::fs::write(
        out_dir.join("report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    Ok(())
}
