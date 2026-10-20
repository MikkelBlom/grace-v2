//! `grace-chat-bench`: scripted conversations through Grace's real system message, scored for
//! conversation style (docs/conversation-style.md).
//!
//!   grace-chat-bench --label baseline [--only 03] [--seeds 3] [--no-judge]
//!
//! Each scenario in `fixtures/chat/scenarios/` is a list of user turns, scripted or played by a
//! simulated user (Gemma role-playing a fictional Danish user, fixed seed). Grace answers through
//! `conductor::system_message` with the scenario's memory rendered by `memory::session_block`,
//! at the engine's sampling settings. Every reply gets hard checks in code (length, questions,
//! banned phrases, silence, speakable text); then a local LLM judge scores the whole transcript
//! against the scenario's rubric. Needs llama-server only (scripts/grace.ps1 starts it).
//! Results: `bench-results/chat/<label>/` (report.json, one transcript per scenario and seed).

use std::{collections::BTreeMap, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use grace::{conductor::system_message, memory};
use grace_llm::client::{ChatRequest, LlmClient};
use grace_memory::{Fact, FactKind};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};

const LLM_URL: &str = "http://127.0.0.1:8080";
/// The engine's reply settings (conductor::speak_reply): ChatRequest's default temperature.
const REPLY_MAX_TOKENS: u32 = 300;
/// A fixed "now" so the date line in the system message never changes between runs.
const NOW: &str = "2026-10-21T14:10:00+02:00";

/// Said by a friend, never by Grace (docs/conversation-style.md). Case-insensitive substrings,
/// or `re:` regexes.
const BANNED: &[&str] = &[
    "certainly",
    "great question",
    "here to help",
    "as an ai",
    "noted that down",
    "i'll keep quiet",
    "i'll stay quiet",
    "i'll be quiet",
    "staying quiet",
    "no worries",
    "if that makes sense",
    "i get that",
    "that sounds like",
    "i'm guessing",
    "re:\\bc#",
    "re:\\bf#",
    "c\\+\\+",
];
/// Danish words that are not English words; two or more in a reply means it's in Danish.
const DANISH_WORDS: &[&str] = &[
    "jeg", "ikke", "det", "og", "har", "du", "til", "af", "kan", "vil", "skal", "hvad", "så", "på",
    "er", "med", "mig", "dig", "der", "være", "også", "bare", "lige", "nej", "jo",
];
/// What counts as staying quiet: nothing, or one of these.
const QUIET_OK: &[&str] = &["", "okay", "ok", "mm", "mhm", "sure", "got it", "alright"];

#[derive(Parser)]
struct Args {
    #[arg(long)]
    label: String,
    #[arg(long, default_value = "fixtures/chat/scenarios")]
    scenarios: PathBuf,
    #[arg(long, default_value = "fixtures/chat/persona.md")]
    persona: PathBuf,
    #[arg(long, default_value = "docs/conversation-style.md")]
    style: PathBuf,
    #[arg(long, default_value = "config/prompts/system.md")]
    system_prompt: PathBuf,
    /// Only scenarios whose id starts with this.
    #[arg(long)]
    only: Option<String>,
    /// Runs per scenario (seeds 1..=n), so one lucky sample doesn't decide a prompt change.
    #[arg(long, default_value_t = 3)]
    seeds: u64,
    /// Hard checks only.
    #[arg(long)]
    no_judge: bool,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    #[serde(default)]
    categories: Vec<String>,
    /// What the conversation is about, for the simulated user and the judge.
    intent: String,
    /// Facts already in Grace's memory, rendered exactly as the engine does.
    #[serde(default)]
    memory: Vec<String>,
    /// Checks applied to every reply, merged under each turn's own.
    #[serde(default)]
    every: Expect,
    turns: Vec<Turn>,
    /// Upper bound on the share of replies that end with a question.
    #[serde(default)]
    question_end_rate_max: Option<f64>,
    /// Rubric for the judge: yes/no statements that should be true of Grace's replies.
    #[serde(default)]
    judge: Vec<String>,
}

#[derive(Deserialize)]
struct Turn {
    /// A scripted user line (Danish, as speech recognition writes it).
    #[serde(default)]
    user: Option<String>,
    /// Or: what the simulated user does this turn.
    #[serde(default)]
    sim: Option<String>,
    /// A fixed Grace reply that sets up context (not generated, not checked).
    #[serde(default)]
    grace: Option<String>,
    #[serde(default)]
    expect: Expect,
}

#[derive(Deserialize, Default, Clone)]
struct Expect {
    max_words: Option<usize>,
    min_words: Option<usize>,
    max_sentences: Option<usize>,
    max_questions: Option<usize>,
    /// Reply must not contain a question.
    #[serde(default)]
    no_question: bool,
    /// Reply must contain a question.
    #[serde(default)]
    question: bool,
    /// Reply must be empty or a bare acknowledgement (QUIET_OK).
    #[serde(default)]
    quiet: bool,
    /// Reply must not re-ask the question of Grace's previous reply.
    #[serde(default)]
    no_repeat: bool,
    #[serde(default)]
    must_match: Vec<String>,
    #[serde(default)]
    must_not_match: Vec<String>,
}

impl Expect {
    fn merged(&self, every: &Expect) -> Expect {
        let mut m = self.clone();
        m.max_words = m.max_words.or(every.max_words);
        m.min_words = m.min_words.or(every.min_words);
        m.max_sentences = m.max_sentences.or(every.max_sentences);
        m.max_questions = m.max_questions.or(every.max_questions);
        m.no_question |= every.no_question;
        m.question |= every.question;
        m.quiet |= every.quiet;
        m.no_repeat |= every.no_repeat;
        m.must_match.extend(every.must_match.iter().cloned());
        m.must_not_match
            .extend(every.must_not_match.iter().cloned());
        m
    }
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

fn sentences(text: &str) -> usize {
    text.split(['.', '!', '?'])
        .filter(|s| s.chars().any(char::is_alphanumeric))
        .count()
}

/// Words that make every question look alike ("Who do you live with?" vs "What do you work with?").
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "any", "are", "at", "do", "does", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "so", "the", "to", "what", "what's", "when", "where", "who", "why", "with",
    "you", "your", "you're",
];

/// The content words of the last question in a reply, lowercased.
fn last_question(text: &str) -> Option<String> {
    let end = text.rfind('?')?;
    let start = text[..end]
        .rfind(['.', '!', '?', '\n'])
        .map_or(0, |i| i + 1);
    Some(
        text[start..end]
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty() && !STOPWORDS.contains(w))
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// Share of the shorter question's words that also appear in the other.
fn overlap(a: &str, b: &str) -> f64 {
    let (a, b): (Vec<&str>, Vec<&str>) = (a.split(' ').collect(), b.split(' ').collect());
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.is_empty() {
        return 0.0;
    }
    short.iter().filter(|w| long.contains(w)).count() as f64 / short.len() as f64
}

fn is_quiet(reply: &str) -> bool {
    let t: String = reply
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    QUIET_OK.contains(&t.trim())
}

/// Hard checks on one reply: (name, pass, detail).
fn hard_checks(e: &Expect, reply: &str, previous: Option<&str>) -> Vec<(String, bool, String)> {
    let mut out = Vec::new();
    let mut add =
        |name: &str, pass: bool, detail: String| out.push((name.to_owned(), pass, detail));
    let q = reply.matches('?').count();
    if let Some(m) = e.max_words {
        add(
            "max_words",
            words(reply) <= m,
            format!("{} > {m}", words(reply)),
        );
    }
    if let Some(m) = e.min_words {
        add(
            "min_words",
            words(reply) >= m,
            format!("{} < {m}", words(reply)),
        );
    }
    if let Some(m) = e.max_sentences {
        add(
            "max_sentences",
            sentences(reply) <= m,
            format!("{} > {m}", sentences(reply)),
        );
    }
    if let Some(m) = e.max_questions {
        add("max_questions", q <= m, format!("{q} > {m}"));
    }
    if e.no_question {
        add("no_question", q == 0, format!("{q} questions"));
    }
    if e.question {
        add("question", q > 0, "no question".into());
    }
    if e.quiet {
        add("quiet", is_quiet(reply), "spoke".into());
    }
    if e.no_repeat
        && let (Some(now), Some(before)) = (last_question(reply), previous.and_then(last_question))
    {
        let o = overlap(&now, &before);
        add(
            "no_repeat",
            o < 0.6,
            format!("{:.0} % same words as last question", o * 100.0),
        );
    }
    for m in &e.must_match {
        add("must_match", matches(m, reply), m.clone());
    }
    for m in &e.must_not_match {
        add("must_not_match", !matches(m, reply), m.clone());
    }
    // Every reply: spoken-style text, no banned phrases.
    let banned: Vec<&str> = BANNED
        .iter()
        .copied()
        .filter(|b| matches(b, reply))
        .collect();
    add("banned", banned.is_empty(), banned.join(", "));
    // Grace always answers in English (ADR 0011). Danish names and places stay, so look for
    // Danish function words, which never appear in an English sentence.
    let lower = reply.to_lowercase();
    let danish: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| DANISH_WORDS.contains(w))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    add(
        "english",
        danish.len() < 2,
        format!("Danish: {}", danish.join(" ")),
    );
    // Exclamation marks are the cheapest tell of trying too hard.
    add(
        "no_exclamation",
        !reply.contains('!'),
        "exclamation mark".into(),
    );
    // "<|channel>" is Gemma's thinking channel leaking out (seen when asked to stay silent).
    let markup = Regex::new(r"(?m)(\*|^#|^\s*[-•]\s|[\u{1F300}-\u{1FAFF}]|https?://|<\|)")
        .expect("regex")
        .is_match(reply);
    add(
        "speakable",
        !markup,
        "markdown, emoji, URL or model tokens".into(),
    );
    out
}

struct Check {
    scenario: String,
    seed: u64,
    turn: usize,
    kind: &'static str,
    name: String,
    pass: bool,
    detail: String,
}

struct Bench {
    args: Args,
    llm: LlmClient,
    persona: String,
    style: String,
    checks: Vec<Check>,
    out: PathBuf,
    /// Every generated exchange, for tools that compare runs (bench/style_pick.py).
    exchanges: Vec<Value>,
}

impl Bench {
    async fn grace(
        &self,
        system: &str,
        history: &[(&'static str, String)],
        seed: u64,
    ) -> anyhow::Result<String> {
        let mut messages = vec![json!({ "role": "system", "content": system })];
        messages.extend(
            history
                .iter()
                .map(|(r, t)| json!({ "role": r, "content": t })),
        );
        let mut req = ChatRequest::new(messages);
        req.max_tokens = REPLY_MAX_TOKENS;
        req.extra.insert("seed".into(), json!(seed));
        Ok(self.llm.complete(&req).await?.trim().to_owned())
    }

    /// The simulated user's next line: Gemma playing the persona, in Danish, as STT writes it.
    async fn sim_user(
        &self,
        sc: &Scenario,
        goal: &str,
        history: &[(&'static str, String)],
        seed: u64,
    ) -> anyhow::Result<String> {
        let system = format!(
            "You are role-playing a person talking out loud to Grace, his voice assistant. Stay in \
             character; you are not an assistant.\n\n{}\n\nThe conversation: {}\n\nWhat you do \
             now: {goal}\n\nWrite only your next spoken line, in casual spoken Danish, the way \
             speech recognition writes it: no quotes, no stage directions, no emoji, usually one or \
             two short sentences, sometimes with 'øh'. Grace speaks English; you understand it.",
            self.persona, sc.intent
        );
        // Roles flipped: the sim user's lines are its own (assistant) turns.
        let mut messages = vec![json!({ "role": "system", "content": system })];
        let flipped: Vec<Value> = history
            .iter()
            .map(|(r, t)| {
                let role = if *r == "user" { "assistant" } else { "user" };
                json!({ "role": role, "content": if t.is_empty() { "(Grace says nothing.)" } else { t } })
            })
            .collect();
        if flipped.first().is_some_and(|m| m["role"] == "assistant") {
            messages.push(json!({ "role": "user", "content": "(Grace is listening.)" }));
        }
        messages.extend(flipped);
        if messages.last().is_some_and(|m| m["role"] == "assistant") {
            messages.push(json!({ "role": "user", "content": "(Grace says nothing.)" }));
        }
        let mut req = ChatRequest::new(messages);
        req.max_tokens = 120;
        req.temperature = 0.8;
        req.extra.insert("seed".into(), json!(seed * 1000 + 7));
        let line = self.llm.complete(&req).await?;
        Ok(line.trim().trim_matches('"').to_owned())
    }

    async fn judge(
        &self,
        sc: &Scenario,
        transcript: &str,
    ) -> anyhow::Result<Vec<(String, bool, String)>> {
        let general = [
            "Grace sounds like a capable human talking: brief, dry and competent, not a customer-service bot, a therapist or a generic AI assistant.",
            "Grace does not mirror or paraphrase Mikkel's feelings back to him therapist-style ('so you're ...', 'it's like you want ...').",
            "Grace is not trying too hard: no forced jokes, no quip in every reply, no puns, no exclamation marks; most replies are plain and short, with wit as an occasional light touch.",
        ];
        let criteria: Vec<&str> = sc.judge.iter().map(String::as_str).chain(general).collect();
        let list: String = criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {c}\n", i + 1))
            .collect();
        let system = format!(
            "You grade conversations between Mikkel (speaking Danish through speech recognition) \
             and Grace, his English-speaking voice assistant. Everything Grace writes is read \
             aloud. The style Grace should have:\n\n{}\n\nBe strict: a criterion passes only if it \
             holds for every relevant Grace reply.",
            self.style
        );
        let user = format!(
            "Scenario: {}\n\nTranscript:\n{transcript}\n\nFor each criterion, give a one-sentence \
             reason quoting Grace, then pass true or false.\n\nCriteria:\n{list}",
            sc.intent
        );
        let schema = json!({
            "type": "object",
            "properties": { "criteria": { "type": "array", "minItems": criteria.len(), "maxItems": criteria.len(), "items": {
                "type": "object",
                "properties": { "reason": { "type": "string" }, "pass": { "type": "boolean" } },
                "required": ["reason", "pass"], "additionalProperties": false } } },
            "required": ["criteria"], "additionalProperties": false
        });
        let mut req = ChatRequest::new(vec![
            json!({ "role": "system", "content": system }),
            json!({ "role": "user", "content": user }),
        ]);
        req.max_tokens = 1500;
        req.temperature = 0.0;
        req.extra.insert("seed".into(), json!(1));
        let v = self.llm.complete_json(&req, &schema).await?;
        let items = v["criteria"].as_array().context("judge: no criteria")?;
        Ok(criteria
            .iter()
            .zip(items)
            .map(|(c, it)| {
                (
                    (*c).to_owned(),
                    it["pass"].as_bool().unwrap_or(false),
                    it["reason"].as_str().unwrap_or("").to_owned(),
                )
            })
            .collect())
    }

    async fn run(&mut self, sc: &Scenario, seed: u64) -> anyhow::Result<()> {
        let now = chrono::DateTime::parse_from_rfc3339(NOW)?.with_timezone(&chrono::Local);
        let facts: Vec<Fact> = sc
            .memory
            .iter()
            .enumerate()
            .map(|(i, t)| Fact {
                id: i as i64 + 1,
                text: t.clone(),
                kind: FactKind::User,
                confidence: 0.9,
                source_turn: None,
                recorded_at: 0.0,
            })
            .collect();
        let block = memory::session_block(&facts, &[]);
        let system = system_message(&self.args.system_prompt, now, &block);
        let mut history: Vec<(&'static str, String)> = Vec::new();
        let mut transcript = String::new();
        let mut previous: Option<String> = None;
        let (mut replies, mut q_end) = (0usize, 0usize);
        for (i, turn) in sc.turns.iter().enumerate() {
            let user = match (&turn.user, &turn.sim) {
                (Some(u), _) => u.clone(),
                (None, Some(goal)) => {
                    self.sim_user(sc, goal, &history, seed * 100 + i as u64)
                        .await?
                }
                _ => anyhow::bail!("turn {i} has neither user nor sim"),
            };
            // Like the engine: an empty reply is never spoken and never enters the history, so
            // two user turns can follow each other.
            history.push(("user", user.clone()));
            if let Some(fixed) = &turn.grace {
                transcript.push_str(&format!(
                    "[{}] Mikkel: {user}\n[{}] Grace (scripted): {fixed}\n",
                    i + 1,
                    i + 1
                ));
                history.push(("assistant", fixed.clone()));
                previous = Some(fixed.clone());
                continue;
            }
            let reply = match self.grace(&system, &history, seed * 100 + i as u64).await {
                Ok(r) => r,
                Err(e) => {
                    self.record(sc, seed, i, "hard", "llm_error", false, format!("{e:#}"));
                    String::new()
                }
            };
            self.exchanges.push(json!({
                "scenario": sc.id, "seed": seed, "turn": i + 1, "user": user,
                "before": previous, "reply": reply, "simulated": turn.sim.is_some(),
            }));
            transcript.push_str(&format!(
                "[{}] Mikkel{}: {user}\n[{}] Grace: {}\n",
                i + 1,
                if turn.sim.is_some() {
                    " (simulated)"
                } else {
                    ""
                },
                i + 1,
                if reply.is_empty() { "(silent)" } else { &reply }
            ));
            let expect = turn.expect.merged(&sc.every);
            for (name, pass, detail) in hard_checks(&expect, &reply, previous.as_deref()) {
                let detail = if pass {
                    String::new()
                } else {
                    format!("{detail} :: {reply}")
                };
                self.record(sc, seed, i + 1, "hard", &name, pass, detail);
            }
            if !reply.is_empty() {
                replies += 1;
                q_end += reply.trim_end().ends_with('?') as usize;
                history.push(("assistant", reply.clone()));
                previous = Some(reply);
            }
        }
        if let Some(max) = sc.question_end_rate_max {
            let rate = q_end as f64 / replies.max(1) as f64;
            self.record(
                sc,
                seed,
                0,
                "hard",
                "question_end_rate",
                rate <= max,
                format!("{q_end}/{replies} > {max}"),
            );
        }
        if !self.args.no_judge {
            match self.judge(sc, &transcript).await {
                Ok(verdicts) => {
                    for (c, pass, reason) in verdicts {
                        self.record(sc, seed, 0, "judge", &c, pass, reason);
                    }
                }
                Err(e) => self.record(sc, seed, 0, "judge", "judge_error", false, format!("{e:#}")),
            }
        }
        let fails: String = self
            .checks
            .iter()
            .filter(|c| c.scenario == sc.id && c.seed == seed && !c.pass)
            .map(|c| format!("- [{}] t{} {}: {}\n", c.kind, c.turn, c.name, c.detail))
            .collect();
        std::fs::write(
            self.out
                .join("transcripts")
                .join(format!("{}-s{seed}.md", sc.id)),
            format!(
                "# {} (seed {seed})\n\n{}\n\n{transcript}\n## Failed\n{fails}",
                sc.id, sc.intent
            ),
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        sc: &Scenario,
        seed: u64,
        turn: usize,
        kind: &'static str,
        name: &str,
        pass: bool,
        detail: String,
    ) {
        self.checks.push(Check {
            scenario: sc.id.clone(),
            seed,
            turn,
            kind,
            name: name.to_owned(),
            pass,
            detail,
        });
    }
}

fn rate(checks: &[&Check]) -> (usize, usize) {
    (checks.iter().filter(|c| c.pass).count(), checks.len())
}

fn frac((p, n): (usize, usize)) -> Value {
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
    let mut files: Vec<PathBuf> = std::fs::read_dir(&args.scenarios)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
        .collect();
    files.sort();
    let mut scenarios = Vec::new();
    for f in files {
        // A file holds one scenario, or a list of them (short single exchanges).
        let text = std::fs::read_to_string(&f)?;
        let many = match serde_yaml::from_str::<Vec<Scenario>>(&text) {
            Ok(v) => v,
            Err(_) => vec![
                serde_yaml::from_str::<Scenario>(&text)
                    .with_context(|| format!("parsing {}", f.display()))?,
            ],
        };
        for sc in many {
            if args.only.as_deref().is_none_or(|o| sc.id.starts_with(o)) {
                scenarios.push(sc);
            }
        }
    }
    let out = PathBuf::from(format!("bench-results/chat/{}", args.label));
    std::fs::create_dir_all(out.join("transcripts"))?;
    // The prompt this run measured, for before/after comparisons.
    std::fs::copy(&args.system_prompt, out.join("system.md"))?;
    let mut bench = Bench {
        persona: std::fs::read_to_string(&args.persona).context("persona")?,
        style: std::fs::read_to_string(&args.style).context("style spec")?,
        args,
        llm,
        checks: Vec::new(),
        out: out.clone(),
        exchanges: Vec::new(),
    };
    let started = std::time::Instant::now();
    for sc in &scenarios {
        for seed in 1..=bench.args.seeds {
            if let Err(e) = bench.run(sc, seed).await {
                bench.record(sc, seed, 0, "hard", "error", false, format!("{e:#}"));
            }
        }
        let mine: Vec<&Check> = bench
            .checks
            .iter()
            .filter(|c| c.scenario == sc.id)
            .collect();
        let hard = rate(
            &mine
                .iter()
                .copied()
                .filter(|c| c.kind == "hard")
                .collect::<Vec<_>>(),
        );
        let judge = rate(
            &mine
                .iter()
                .copied()
                .filter(|c| c.kind == "judge")
                .collect::<Vec<_>>(),
        );
        eprintln!(
            "{:<32} hard {:>3}/{:<3} judge {:>2}/{:<2}",
            sc.id, hard.0, hard.1, judge.0, judge.1
        );
    }
    let all: Vec<&Check> = bench.checks.iter().collect();
    let of =
        |kind: &str| -> Vec<&Check> { all.iter().copied().filter(|c| c.kind == kind).collect() };
    let (hard, judge) = (rate(&of("hard")), rate(&of("judge")));
    let pct = |(p, n): (usize, usize)| p as f64 / n.max(1) as f64;
    // Score = mean of hard-check and judge pass rates (each 0..1), so neither dominates by count.
    let score = if bench.args.no_judge {
        pct(hard)
    } else {
        (pct(hard) + pct(judge)) / 2.0
    };
    let mut by_check: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut by_scenario: BTreeMap<String, Value> = BTreeMap::new();
    for c in &all {
        if c.kind == "hard" {
            let e = by_check.entry(c.name.clone()).or_default();
            e.0 += c.pass as usize;
            e.1 += 1;
        }
    }
    for sc in &scenarios {
        let mine: Vec<&Check> = all
            .iter()
            .copied()
            .filter(|c| c.scenario == sc.id)
            .collect();
        let h = rate(
            &mine
                .iter()
                .copied()
                .filter(|c| c.kind == "hard")
                .collect::<Vec<_>>(),
        );
        let j = rate(
            &mine
                .iter()
                .copied()
                .filter(|c| c.kind == "judge")
                .collect::<Vec<_>>(),
        );
        by_scenario.insert(
            sc.id.clone(),
            json!({ "hard": frac(h), "judge": frac(j), "categories": sc.categories }),
        );
    }
    let summary = json!({
        "label": bench.args.label,
        "system_prompt": bench.args.system_prompt,
        "scenarios": scenarios.len(),
        "seeds": bench.args.seeds,
        "hard": frac(hard),
        "judge": frac(judge),
        "score": (score * 1000.0).round() / 1000.0,
        "by_check": by_check.into_iter().map(|(k, v)| (k, frac(v))).collect::<serde_json::Map<_, _>>(),
        "by_scenario": by_scenario,
        "wall_s": started.elapsed().as_secs_f64().round(),
    });
    let items: Vec<Value> = all
        .iter()
        .map(|c| json!({"scenario": c.scenario, "seed": c.seed, "turn": c.turn, "kind": c.kind, "name": c.name, "pass": c.pass, "detail": c.detail}))
        .collect();
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&json!({"summary": summary, "checks": items}))?,
    )?;
    let lines: Vec<String> = bench.exchanges.iter().map(Value::to_string).collect();
    std::fs::write(out.join("transcripts.jsonl"), lines.join("\n") + "\n")?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
