//! `grace-style-bench`: does Grace learn how Mikkel wants her to talk, and keep it (ADR 0017)?
//!
//!   grace-style-bench --label v1 [--only 01] [--seeds 3] [--mode both|on|off] [--no-judge]
//!
//! Each scenario in `fixtures/style/` is one or more scripted sessions of Danish user lines.
//! Grace answers through `conductor::system_message` at the engine's sampling settings, with the
//! style rules rendered into it by `style::render_rules`. With learning on, every exchange then
//! runs the real `style::learn` against a scratch in-memory database, so a rule learned on one
//! turn shapes the next one and survives into the next session (history cleared, rules kept).
//! With learning off (the baseline: feedback lives only in the chat history, as in live run 4)
//! the same lines are replayed without rules.
//!
//! Measured per mode: mean reply words and question rate on turns tagged `before` (before the
//! feedback), `after` (same session) and `next` (a later session); the classifier's rule
//! changes on turns labelled `feedback` vs on all other turns (false rules); end-state rule
//! checks; per-turn hard checks; and an optional LLM judge (e.g. "rules don't pile up into
//! overdone behaviour"). Needs llama-server only. Results: `bench-results/style/<label>/`.

use std::{collections::BTreeMap, path::PathBuf, sync::Mutex};

use anyhow::Context;
use clap::Parser;
use grace::{conductor::system_message, memory, style};
use grace_llm::client::{ChatRequest, LlmClient};
use grace_memory::Memory;
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};

const LLM_URL: &str = "http://127.0.0.1:8080";
/// The engine's reply settings (conductor::speak_reply).
const REPLY_MAX_TOKENS: u32 = 300;
const NOW: &str = "2026-10-21T14:10:00+02:00";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    label: String,
    #[arg(long, default_value = "fixtures/style")]
    scenarios: PathBuf,
    #[arg(long, default_value = "config/prompts/system.md")]
    system_prompt: PathBuf,
    #[arg(long)]
    only: Option<String>,
    #[arg(long, default_value_t = 3)]
    seeds: u64,
    /// `on` (rules learned and applied), `off` (baseline), or `both`.
    #[arg(long, default_value = "both")]
    mode: String,
    #[arg(long)]
    no_judge: bool,
    /// Don't restate the rules on the current user message (the engine does, `style::with_reminder`).
    #[arg(long)]
    no_remind: bool,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    intent: String,
    /// Rules already learned before the scenario starts: (rule, his words).
    #[serde(default)]
    rules: Vec<PresetRule>,
    sessions: Vec<Session>,
    /// End state (learning on): every inner list needs one current rule containing one of its
    /// words.
    #[serde(default)]
    rules_match: Vec<Vec<String>>,
    #[serde(default)]
    rules_count: Option<usize>,
    /// Yes/no statements about Grace's replies, judged on the whole transcript.
    #[serde(default)]
    judge: Vec<String>,
}

#[derive(Deserialize)]
struct PresetRule {
    rule: String,
    said: String,
}

#[derive(Deserialize)]
struct Session {
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    user: String,
    /// A fixed Grace reply (e.g. a joke he then praises), not generated or measured.
    #[serde(default)]
    grace: Option<String>,
    /// Ground truth: this line is feedback that should add, change or remove a rule.
    #[serde(default)]
    feedback: bool,
    /// `before`, `after` or `next`: which group this reply counts in.
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    expect: Expect,
}

#[derive(Deserialize, Default)]
struct Expect {
    max_words: Option<usize>,
    min_words: Option<usize>,
    max_questions: Option<usize>,
    #[serde(default)]
    must_match: Vec<String>,
    #[serde(default)]
    must_not_match: Vec<String>,
}

fn matches(pattern: &str, text: &str) -> bool {
    match pattern.strip_prefix("re:") {
        Some(re) => Regex::new(&format!("(?i){re}")).is_ok_and(|r| r.is_match(text)),
        None => text.to_lowercase().contains(&pattern.to_lowercase()),
    }
}

fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

#[derive(Default, Clone, Copy)]
struct Group {
    replies: usize,
    words: usize,
    questions: usize,
}

impl Group {
    fn add(&mut self, reply: &str) {
        self.replies += 1;
        self.words += words(reply);
        self.questions += reply.contains('?') as usize;
    }
    fn json(&self) -> Value {
        let n = self.replies.max(1) as f64;
        json!({
            "replies": self.replies,
            "mean_words": (self.words as f64 / n * 10.0).round() / 10.0,
            "question_rate": (self.questions as f64 / n * 100.0).round() / 100.0,
        })
    }
}

struct Check {
    scenario: String,
    mode: &'static str,
    seed: u64,
    turn: String,
    name: String,
    pass: bool,
    detail: String,
}

#[derive(Default)]
struct Classifier {
    /// Feedback turns where a rule changed / all feedback turns.
    hit: usize,
    feedback_turns: usize,
    /// Other turns where a rule changed anyway / all other turns.
    false_rules: usize,
    other_turns: usize,
    reviews: usize,
}

struct Bench {
    args: Args,
    llm: LlmClient,
    checks: Vec<Check>,
    /// (scenario, mode, tag) → replies
    groups: BTreeMap<(String, &'static str, String), Group>,
    classifier: Classifier,
    out: PathBuf,
}

impl Bench {
    async fn grace(
        &self,
        system: &str,
        history: &[(&'static str, String)],
        rules: &[grace_memory::StyleRule],
        seed: u64,
    ) -> anyhow::Result<String> {
        let mut messages = vec![json!({ "role": "system", "content": system })];
        messages.extend(
            history
                .iter()
                .map(|(r, t)| json!({ "role": r, "content": t })),
        );
        if !self.args.no_remind
            && let Some(last) = messages.last_mut()
            && !rules.is_empty()
        {
            let text = last["content"].as_str().unwrap_or("").to_owned();
            last["content"] = json!(style::with_reminder(&text, rules));
        }
        let mut req = ChatRequest::new(messages);
        req.max_tokens = REPLY_MAX_TOKENS;
        req.extra.insert("seed".into(), json!(seed));
        Ok(self.llm.complete(&req).await?.trim().to_owned())
    }

    async fn judge(
        &self,
        criteria: &[String],
        intent: &str,
        transcript: &str,
    ) -> anyhow::Result<Vec<(bool, String)>> {
        let list: String = criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {c}\n", i + 1))
            .collect();
        let schema = json!({
            "type": "object",
            "properties": { "criteria": { "type": "array", "minItems": criteria.len(), "maxItems": criteria.len(), "items": {
                "type": "object",
                "properties": { "reason": { "type": "string" }, "pass": { "type": "boolean" } },
                "required": ["reason", "pass"], "additionalProperties": false } } },
            "required": ["criteria"], "additionalProperties": false
        });
        let mut req = ChatRequest::new(vec![
            json!({ "role": "system", "content": "You grade conversations between Mikkel (speaking Danish through speech recognition) and Grace, his English-speaking voice assistant, whose replies are read aloud. Be strict: a criterion passes only if it holds for every relevant Grace reply." }),
            json!({ "role": "user", "content": format!("Scenario: {intent}\n\nTranscript:\n{transcript}\n\nFor each criterion give a one-sentence reason quoting Grace, then pass true or false.\n\nCriteria:\n{list}") }),
        ]);
        req.max_tokens = 1200;
        req.temperature = 0.0;
        req.extra.insert("seed".into(), json!(1));
        let v = self.llm.complete_json(&req, &schema).await?;
        let items = v["criteria"].as_array().context("judge: no criteria")?;
        Ok(items
            .iter()
            .map(|it| {
                (
                    it["pass"].as_bool().unwrap_or(false),
                    it["reason"].as_str().unwrap_or("").to_owned(),
                )
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        sc: &str,
        mode: &'static str,
        seed: u64,
        turn: &str,
        name: &str,
        pass: bool,
        detail: String,
    ) {
        self.checks.push(Check {
            scenario: sc.to_owned(),
            mode,
            seed,
            turn: turn.to_owned(),
            name: name.to_owned(),
            pass,
            detail,
        });
    }

    async fn run(&mut self, sc: &Scenario, mode: &'static str, seed: u64) -> anyhow::Result<()> {
        let learning = mode == "on";
        let now = chrono::DateTime::parse_from_rfc3339(NOW)?.with_timezone(&chrono::Local);
        let db = Mutex::new(Memory::open_in_memory(memory::EMBEDDER, memory::DIM)?);
        if learning {
            let db = db.lock().expect("db");
            for (i, r) in sc.rules.iter().enumerate() {
                db.add_style_rule(&r.rule, &r.said, None, None, i as f64)?;
            }
        }
        let memory_block = memory::session_block(&[], &[]);
        let mut transcript = String::new();
        let mut clock = 1000.0;
        for (si, session) in sc.sessions.iter().enumerate() {
            let sid = db.lock().expect("db").start_session(clock)?;
            transcript.push_str(&format!("--- session {} ---\n", si + 1));
            let mut history: Vec<(&'static str, String)> = Vec::new();
            for (ti, turn) in session.turns.iter().enumerate() {
                let label = format!("s{}t{}", si + 1, ti + 1);
                clock += 30.0;
                let rules = db.lock().expect("db").style_rules()?;
                let block = match style::render_rules(&rules) {
                    r if r.is_empty() => memory_block.clone(),
                    r => format!("{memory_block}\n\n{r}"),
                };
                let system = system_message(&self.args.system_prompt, now, &block);
                let user_turn = db
                    .lock()
                    .expect("db")
                    .add_turn(sid, "user", &turn.user, clock)?;
                history.push(("user", turn.user.clone()));
                let reply = match &turn.grace {
                    Some(fixed) => fixed.clone(),
                    None => match self
                        .grace(
                            &system,
                            &history,
                            &rules,
                            seed * 100 + (si * 20 + ti) as u64,
                        )
                        .await
                    {
                        Ok(r) => r,
                        Err(e) => {
                            self.record(
                                &sc.id,
                                mode,
                                seed,
                                &label,
                                "llm_error",
                                false,
                                format!("{e:#}"),
                            );
                            String::new()
                        }
                    },
                };
                transcript.push_str(&format!(
                    "[{label}] Mikkel{}: {}\n[{label}] Grace{}: {}\n",
                    if turn.feedback { " (feedback)" } else { "" },
                    turn.user,
                    if turn.grace.is_some() {
                        " (scripted)"
                    } else {
                        ""
                    },
                    if reply.is_empty() { "(silent)" } else { &reply }
                ));
                if turn.grace.is_none() {
                    if let Some(tag) = &turn.tag {
                        self.groups
                            .entry((sc.id.clone(), mode, tag.clone()))
                            .or_default()
                            .add(&reply);
                    }
                    let e = &turn.expect;
                    // Checked in both modes: the baseline's pass rate is the "before" number.
                    {
                        let q = reply.matches('?').count();
                        if let Some(m) = e.max_words {
                            self.record(
                                &sc.id,
                                mode,
                                seed,
                                &label,
                                "max_words",
                                words(&reply) <= m,
                                format!("{} > {m} :: {reply}", words(&reply)),
                            );
                        }
                        if let Some(m) = e.min_words {
                            self.record(
                                &sc.id,
                                mode,
                                seed,
                                &label,
                                "min_words",
                                words(&reply) >= m,
                                format!("{} < {m} :: {reply}", words(&reply)),
                            );
                        }
                        if let Some(m) = e.max_questions {
                            self.record(
                                &sc.id,
                                mode,
                                seed,
                                &label,
                                "max_questions",
                                q <= m,
                                format!("{q} > {m} :: {reply}"),
                            );
                        }
                        for p in &e.must_match {
                            self.record(
                                &sc.id,
                                mode,
                                seed,
                                &label,
                                "must_match",
                                matches(p, &reply),
                                format!("{p} :: {reply}"),
                            );
                        }
                        for p in &e.must_not_match {
                            self.record(
                                &sc.id,
                                mode,
                                seed,
                                &label,
                                "must_not_match",
                                !matches(p, &reply),
                                format!("{p} :: {reply}"),
                            );
                        }
                    }
                }
                if !reply.is_empty() {
                    db.lock()
                        .expect("db")
                        .add_turn(sid, "assistant", &reply, clock + 5.0)?;
                    history.push(("assistant", reply.clone()));
                }
                if learning {
                    let (previous, context) = {
                        let d = db.lock().expect("db");
                        (
                            d.reply_before(sid, user_turn)?,
                            d.turns_before(sid, user_turn, style::REVIEW_CONTEXT)?,
                        )
                    };
                    let ex = style::Exchange {
                        session: Some(sid),
                        user_turn: Some(user_turn),
                        user: &turn.user,
                        reply: &reply,
                        previous: previous.as_deref(),
                        context: &context,
                    };
                    let o = style::learn(&self.llm, &db, &ex, clock + 10.0).await?;
                    let changed = o.changed();
                    if o.complaint || changed {
                        self.classifier.reviews += 1;
                    }
                    if turn.feedback {
                        self.classifier.feedback_turns += 1;
                        self.classifier.hit += changed as usize;
                    } else {
                        self.classifier.other_turns += 1;
                        self.classifier.false_rules += changed as usize;
                    }
                    let rules_now: Vec<String> = db
                        .lock()
                        .expect("db")
                        .style_rules()?
                        .into_iter()
                        .map(|r| format!("#{} {}", r.id, r.text))
                        .collect();
                    if changed || !o.refused.is_empty() {
                        transcript.push_str(&format!(
                            "    (rules now: {}{})\n",
                            if rules_now.is_empty() {
                                "none".to_owned()
                            } else {
                                rules_now.join(" | ")
                            },
                            if o.refused.is_empty() {
                                String::new()
                            } else {
                                format!("; refused: {}", o.refused.join("; "))
                            }
                        ));
                    }
                    self.record(
                        &sc.id,
                        mode,
                        seed,
                        &label,
                        "rule_change_as_labelled",
                        changed == turn.feedback,
                        format!(
                            "labelled feedback={} changed={changed} :: rules: {}",
                            turn.feedback,
                            rules_now.join(" | ")
                        ),
                    );
                }
            }
        }
        let rules = db.lock().expect("db").style_rules()?;
        if learning {
            for any in &sc.rules_match {
                let pass = rules
                    .iter()
                    .any(|r| any.iter().any(|w| matches(w, &r.text)));
                let have: Vec<&str> = rules.iter().map(|r| r.text.as_str()).collect();
                self.record(
                    &sc.id,
                    mode,
                    seed,
                    "end",
                    "rules_match",
                    pass,
                    format!("{any:?} not in {have:?}"),
                );
            }
            if let Some(n) = sc.rules_count {
                self.record(
                    &sc.id,
                    mode,
                    seed,
                    "end",
                    "rules_count",
                    rules.len() == n,
                    format!("{} != {n}", rules.len()),
                );
            }
            transcript.push_str(&format!(
                "\nFinal rules: {:?}\n",
                rules.iter().map(|r| &r.text).collect::<Vec<_>>()
            ));
        }
        if !self.args.no_judge && !sc.judge.is_empty() {
            match self.judge(&sc.judge, &sc.intent, &transcript).await {
                Ok(v) => {
                    for (c, (pass, reason)) in sc.judge.iter().zip(v) {
                        self.record(&sc.id, mode, seed, "judge", c, pass, reason);
                    }
                }
                Err(e) => self.record(
                    &sc.id,
                    mode,
                    seed,
                    "judge",
                    "judge_error",
                    false,
                    format!("{e:#}"),
                ),
            }
        }
        let fails: String = self
            .checks
            .iter()
            .filter(|c| c.scenario == sc.id && c.mode == mode && c.seed == seed && !c.pass)
            .map(|c| format!("- {} {}: {}\n", c.turn, c.name, c.detail))
            .collect();
        std::fs::write(
            self.out
                .join("transcripts")
                .join(format!("{}-{mode}-s{seed}.md", sc.id)),
            format!(
                "# {} ({mode}, seed {seed})\n\n{}\n\n{transcript}\n## Failed\n{fails}",
                sc.id, sc.intent
            ),
        )?;
        Ok(())
    }
}

fn frac(p: usize, n: usize) -> Value {
    json!(format!("{p}/{n}"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let llm = LlmClient::new(LLM_URL);
    anyhow::ensure!(
        llm.healthy().await,
        "llama-server is not running at {LLM_URL}"
    );
    let modes: Vec<&'static str> = match args.mode.as_str() {
        "on" => vec!["on"],
        "off" => vec!["off"],
        _ => vec!["off", "on"],
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&args.scenarios)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
        .collect();
    files.sort();
    let mut scenarios = Vec::new();
    for f in files {
        let sc: Scenario = serde_yaml::from_str(&std::fs::read_to_string(&f)?)
            .with_context(|| format!("parsing {}", f.display()))?;
        if args.only.as_deref().is_none_or(|o| sc.id.starts_with(o)) {
            scenarios.push(sc);
        }
    }
    let out = PathBuf::from(format!("bench-results/style/{}", args.label));
    std::fs::create_dir_all(out.join("transcripts"))?;
    std::fs::copy(&args.system_prompt, out.join("system.md"))?;
    let mut bench = Bench {
        args,
        llm,
        checks: Vec::new(),
        groups: BTreeMap::new(),
        classifier: Classifier::default(),
        out: out.clone(),
    };
    let started = std::time::Instant::now();
    for sc in &scenarios {
        for &mode in &modes {
            for seed in 1..=bench.args.seeds {
                if let Err(e) = bench.run(sc, mode, seed).await {
                    bench.record(&sc.id, mode, seed, "-", "error", false, format!("{e:#}"));
                }
            }
            let mine: Vec<&Check> = bench
                .checks
                .iter()
                .filter(|c| c.scenario == sc.id && c.mode == mode)
                .collect();
            eprintln!(
                "{:<28} {:<3} checks {:>3}/{:<3}",
                sc.id,
                mode,
                mine.iter().filter(|c| c.pass).count(),
                mine.len()
            );
        }
    }
    let mut by_mode = serde_json::Map::new();
    for &mode in &modes {
        let mine: Vec<&Check> = bench.checks.iter().filter(|c| c.mode == mode).collect();
        let mut tags: BTreeMap<String, Group> = BTreeMap::new();
        let mut per_scenario = serde_json::Map::new();
        for ((sc, m, tag), g) in &bench.groups {
            if *m != mode {
                continue;
            }
            let t = tags.entry(tag.clone()).or_default();
            t.replies += g.replies;
            t.words += g.words;
            t.questions += g.questions;
            per_scenario
                .entry(sc.clone())
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .expect("object")
                .insert(tag.clone(), g.json());
        }
        let mut by_check: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for c in &mine {
            let name = if c.turn == "judge" {
                "judge".to_owned()
            } else {
                c.name.clone()
            };
            let e = by_check.entry(name).or_default();
            e.0 += c.pass as usize;
            e.1 += 1;
        }
        by_mode.insert(
            mode.to_owned(),
            json!({
                "checks": frac(mine.iter().filter(|c| c.pass).count(), mine.len()),
                "by_check": by_check.into_iter().map(|(k, (p, n))| (k, frac(p, n))).collect::<serde_json::Map<_, _>>(),
                "groups": tags.into_iter().map(|(k, g)| (k, g.json())).collect::<serde_json::Map<_, _>>(),
                "by_scenario": per_scenario,
            }),
        );
    }
    let c = &bench.classifier;
    let summary = json!({
        "label": bench.args.label,
        "system_prompt": bench.args.system_prompt,
        "scenarios": scenarios.len(),
        "seeds": bench.args.seeds,
        "remind": !bench.args.no_remind,
        "modes": by_mode,
        "classifier": {
            "feedback_turns_with_rule_change": frac(c.hit, c.feedback_turns),
            "other_turns_with_rule_change": frac(c.false_rules, c.other_turns),
            "queued_for_review": c.reviews,
        },
        "wall_s": started.elapsed().as_secs_f64().round(),
    });
    let items: Vec<Value> = bench
        .checks
        .iter()
        .map(|c| json!({"scenario": c.scenario, "mode": c.mode, "seed": c.seed, "turn": c.turn, "name": c.name, "pass": c.pass, "detail": c.detail}))
        .collect();
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&json!({"summary": summary, "checks": items}))?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
