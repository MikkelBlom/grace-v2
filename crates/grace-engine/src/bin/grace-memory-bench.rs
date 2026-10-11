//! `grace-memory-bench`: replays the Danish multi-session scenarios in
//! `fixtures/memory/scenarios/` through Grace's real memory path with a fake clock, then checks
//! the memory state and asks probe questions exactly as Grace would be asked.
//!
//!   grace-memory-bench --label baseline [--only 01] [--no-stt-gate]
//!
//! Needs llama-server and the embed server (scripts/grace.ps1 starts both). Every scenario starts
//! from an empty database under `bench-results/memory/<label>/db/`. The scenario format is in
//! `fixtures/memory/scenarios/README.md`.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::Context;
use clap::Parser;
use grace::{
    conductor::system_message,
    memory::{self, Clock, MemoryService},
};
use grace_llm::client::{ChatRequest, LlmClient};
use grace_memory::{Fact, FactKind};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;

const LLM_URL: &str = "http://127.0.0.1:8080";
const EMBED_ADDR: &str = "127.0.0.1:7812";
/// The engine drops transcripts below this avg_logprob before they reach memory (speech.rs).
const STT_GATE: f64 = -0.55;
const ABSTAIN: &[&str] = &[
    "don't know",
    "do not know",
    "not sure",
    "haven't told",
    "have not told",
    "didn't tell",
    "never told",
    "don't remember",
    "don't have",
    "no record",
    "not mentioned",
    "haven't mentioned",
    "never mentioned",
    "no idea",
    "ved ikke",
    "har ikke fortalt",
];

#[derive(Parser)]
struct Args {
    #[arg(long)]
    label: String,
    #[arg(long, default_value = "fixtures/memory/scenarios")]
    scenarios: PathBuf,
    /// Only scenarios whose id starts with this.
    #[arg(long)]
    only: Option<String>,
    /// Pass garbled transcripts to memory instead of dropping them like the engine does.
    #[arg(long)]
    no_stt_gate: bool,
    #[arg(long, default_value = "config/prompts/system.md")]
    system_prompt: PathBuf,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    categories: Vec<String>,
    sessions: Vec<Session>,
}

#[derive(Deserialize)]
struct Session {
    start: String,
    turns: Vec<Turn>,
    #[serde(default)]
    expect: Option<Expect>,
}

#[derive(Deserialize)]
struct Turn {
    user: String,
    grace: String,
    #[serde(default)]
    gap_s: Option<f64>,
    #[serde(default)]
    interrupted: bool,
    #[serde(default)]
    stt_garbled: bool,
    #[serde(default)]
    stt_logprob: Option<f64>,
}

#[derive(Deserialize, Default)]
struct Expect {
    #[serde(default)]
    facts_present: Vec<StateCheck>,
    #[serde(default)]
    facts_absent: Vec<StateCheck>,
    #[serde(default)]
    facts_invalidated: Vec<StateCheck>,
    #[serde(default)]
    commitments_open: Vec<StateCheck>,
    #[serde(default)]
    commitments_done: Vec<StateCheck>,
    #[serde(default)]
    probe_at: Option<String>,
    #[serde(default)]
    probes: Vec<Probe>,
}

#[derive(Deserialize)]
struct StateCheck {
    assert: String,
    #[serde(rename = "match")]
    matcher: Vec<Group>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default)]
    max_count: Option<usize>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    due: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Group {
    One(String),
    Any(Vec<String>),
}

impl Group {
    fn alternatives(&self) -> Vec<&str> {
        match self {
            Group::One(s) => vec![s.as_str()],
            Group::Any(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

#[derive(Deserialize)]
struct Probe {
    question: String,
    #[serde(default)]
    expect: Vec<Group>,
    #[serde(default)]
    must_not_contain: Vec<String>,
    #[serde(default)]
    abstain: bool,
    #[serde(default)]
    proactive: bool,
}

fn norm(s: &str) -> String {
    s.nfc().collect::<String>().to_lowercase()
}

fn alt_matches(alt: &str, text: &str) -> bool {
    match alt.strip_prefix("re:") {
        Some(re) => Regex::new(&format!("(?i){re}")).is_ok_and(|r| r.is_match(text)),
        None => norm(text).contains(&norm(alt)),
    }
}

/// AND over groups, OR within a group.
fn matches(groups: &[Group], text: &str) -> bool {
    groups
        .iter()
        .all(|g| g.alternatives().iter().any(|a| alt_matches(a, text)))
}

fn excluded(exclude: &[String], text: &str) -> bool {
    exclude.iter().any(|e| alt_matches(e, text))
}

fn unix(iso: &str) -> anyhow::Result<f64> {
    Ok(chrono::DateTime::parse_from_rfc3339(iso)
        .with_context(|| format!("bad time {iso}"))?
        .timestamp() as f64)
}

struct Check {
    scenario: String,
    session: usize,
    kind: &'static str,
    what: String,
    pass: bool,
    detail: String,
}

struct Bench {
    args: Args,
    llm: LlmClient,
    now: Arc<AtomicU64>,
    checks: Vec<Check>,
}

impl Bench {
    fn clock(&self) -> Clock {
        let now = self.now.clone();
        Arc::new(move || f64::from_bits(now.load(Ordering::Relaxed)))
    }

    fn set_now(&self, t: f64) {
        self.now.store(t.to_bits(), Ordering::Relaxed);
    }

    fn record(
        &mut self,
        sc: &Scenario,
        session: usize,
        kind: &'static str,
        what: &str,
        pass: bool,
        detail: String,
    ) {
        self.checks.push(Check {
            scenario: sc.id.clone(),
            session,
            kind,
            what: what.to_owned(),
            pass,
            detail,
        });
    }

    async fn open(&self, db: &Path) -> anyhow::Result<Arc<MemoryService>> {
        MemoryService::open_at(db, EMBED_ADDR, self.llm.clone(), self.clock()).await
    }

    async fn run(&mut self, sc: &Scenario) -> anyhow::Result<()> {
        let db = PathBuf::from(format!(
            "bench-results/memory/{}/db/{}.db",
            self.args.label, sc.id
        ));
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{ext}", db.display()));
        }
        let mut pending: Vec<Probe> = Vec::new();
        let mut pending_session = 0;
        for (i, session) in sc.sessions.iter().enumerate() {
            let start = unix(&session.start)?;
            self.set_now(start);
            let mem = self.open(&db).await?;
            // Probes from the previous session are asked now, before this session's turns.
            for p in std::mem::take(&mut pending) {
                self.probe(sc, pending_session, &mem, &p).await?;
            }
            let mut t = start;
            for turn in &session.turns {
                t += turn.gap_s.unwrap_or(20.0);
                self.set_now(t);
                let logprob =
                    turn.stt_logprob
                        .unwrap_or(if turn.stt_garbled { -0.7 } else { -0.2 });
                if !self.args.no_stt_gate && logprob < STT_GATE {
                    continue; // the engine drops it: nothing is answered or remembered
                }
                let user_turn = mem.add_turn("user", &turn.user)?;
                t += 2.0;
                self.set_now(t);
                mem.add_turn("assistant", &turn.grace)?;
                let reply = if turn.interrupted && turn.grace.is_empty() {
                    "(Grace was interrupted before she could answer.)".to_owned()
                } else {
                    turn.grace.clone()
                };
                mem.clone().learn(user_turn, turn.user.clone(), reply).await;
            }
            if let Some(expect) = &session.expect {
                self.check_state(sc, i, &mem, expect)?;
                pending = expect.probes.iter().map(clone_probe).collect();
                pending_session = i;
                if i + 1 == sc.sessions.len() && !pending.is_empty() {
                    let at = expect
                        .probe_at
                        .as_deref()
                        .context("probes on last session need probe_at")?;
                    self.set_now(unix(at)?);
                    let mem = self.open(&db).await?;
                    for p in std::mem::take(&mut pending) {
                        self.probe(sc, i, &mem, &p).await?;
                    }
                }
            }
        }
        Ok(())
    }

    fn check_state(
        &mut self,
        sc: &Scenario,
        i: usize,
        mem: &MemoryService,
        e: &Expect,
    ) -> anyhow::Result<()> {
        let current = mem.facts()?;
        let closed = mem.closed_facts()?;
        let now = f64::from_bits(self.now.load(Ordering::Relaxed));
        let hits = |c: &StateCheck, facts: &[Fact]| -> Vec<String> {
            facts
                .iter()
                .filter(|f| matches(&c.matcher, &f.text) && !excluded(&c.exclude, &f.text))
                .map(|f| f.text.clone())
                .collect()
        };
        let show = |v: &[Fact]| {
            v.iter()
                .map(|f| f.text.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        };
        for c in &e.facts_present {
            let h = hits(c, &current);
            let pass = !h.is_empty() && c.max_count.is_none_or(|m| h.len() <= m);
            let detail = if h.is_empty() {
                format!("current: {}", show(&current))
            } else {
                h.join(" | ")
            };
            self.record(sc, i, "present", &c.assert, pass, detail);
        }
        for c in &e.facts_absent {
            let mut h = hits(c, &current);
            if c.scope.as_deref() == Some("ever") {
                h.extend(hits(c, &closed));
            }
            self.record(sc, i, "absent", &c.assert, h.is_empty(), h.join(" | "));
        }
        for c in &e.facts_invalidated {
            let (open, gone) = (hits(c, &current), hits(c, &closed));
            let pass = open.is_empty() && !gone.is_empty();
            self.record(
                sc,
                i,
                "invalidated",
                &c.assert,
                pass,
                format!(
                    "current: {} / closed: {}",
                    open.join(" | "),
                    gone.join(" | ")
                ),
            );
        }
        let commitments = mem.commitments()?;
        let texts = |open: bool| -> Vec<Fact> {
            commitments
                .iter()
                .filter(|c| c.open == open)
                .map(|c| Fact {
                    id: c.id,
                    text: c.text.clone(),
                    kind: FactKind::User,
                    confidence: 1.0,
                    source_turn: c.source_turn,
                    recorded_at: 0.0,
                })
                .collect()
        };
        let (open_c, closed_c) = (texts(true), texts(false));
        for c in &e.commitments_open {
            let h = hits(c, &open_c);
            let not_due = c.due.as_deref().is_some_and(|d| {
                chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d")
                    .is_ok_and(|d| memory::local_time(now).date_naive() < d)
            });
            let detail = format!(
                "{}{}",
                if h.is_empty() {
                    format!("open: {}", show(&open_c))
                } else {
                    h.join(" | ")
                },
                if not_due { " (not due yet)" } else { "" }
            );
            self.record(sc, i, "commitment_open", &c.assert, !h.is_empty(), detail);
        }
        for c in &e.commitments_done {
            let (open, gone) = (hits(c, &open_c), hits(c, &closed_c));
            self.record(
                sc,
                i,
                "commitment_done",
                &c.assert,
                open.is_empty() && !gone.is_empty(),
                format!("open: {} / closed: {}", open.join(" | "), gone.join(" | ")),
            );
        }
        Ok(())
    }

    async fn probe(
        &mut self,
        sc: &Scenario,
        i: usize,
        mem: &MemoryService,
        p: &Probe,
    ) -> anyhow::Result<()> {
        let now = memory::local_time(f64::from_bits(self.now.load(Ordering::Relaxed)));
        let system = system_message(&self.args.system_prompt, now, mem.session_block());
        let recalled = mem.recall(&p.question).await.unwrap_or_default();
        let messages = vec![
            json!({ "role": "system", "content": system }),
            json!({ "role": "user", "content": memory::with_recall(&p.question, &recalled) }),
        ];
        let mut req = ChatRequest::new(messages);
        req.max_tokens = 300;
        req.temperature = 0.0; // reproducible runs
        let reply = self.llm.complete(&req).await?;
        let mut fails = Vec::new();
        if !p.expect.is_empty() && !matches(&p.expect, &reply) {
            fails.push("missing expected");
        }
        if p.must_not_contain.iter().any(|m| alt_matches(m, &reply)) {
            fails.push("contains forbidden");
        }
        if p.abstain && !ABSTAIN.iter().any(|a| norm(&reply).contains(a)) {
            fails.push("did not abstain");
        }
        let kind = if p.proactive {
            "probe_proactive"
        } else if p.abstain {
            "probe_abstain"
        } else {
            "probe"
        };
        self.record(
            sc,
            i,
            kind,
            &p.question,
            fails.is_empty(),
            format!("{} :: {reply}", fails.join(", ")),
        );
        Ok(())
    }
}

fn clone_probe(p: &Probe) -> Probe {
    Probe {
        question: p.question.clone(),
        expect: p
            .expect
            .iter()
            .map(|g| Group::Any(g.alternatives().iter().map(|s| (*s).to_owned()).collect()))
            .collect(),
        must_not_contain: p.must_not_contain.clone(),
        abstain: p.abstain,
        proactive: p.proactive,
    }
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
        let sc: Scenario = serde_yaml::from_str(&std::fs::read_to_string(&f)?)
            .with_context(|| format!("parsing {}", f.display()))?;
        if args.only.as_deref().is_none_or(|o| sc.id.starts_with(o)) {
            scenarios.push(sc);
        }
    }
    let out = PathBuf::from(format!("bench-results/memory/{}", args.label));
    std::fs::create_dir_all(out.join("db"))?;
    // Every memory operation (learned, skipped and why) lands in a trace next to the report.
    let trace = grace_trace::init("memory-bench", &out)?;
    let mut bench = Bench {
        args,
        llm,
        now: Arc::new(AtomicU64::new(0)),
        checks: Vec::new(),
    };
    let started = std::time::Instant::now();
    for sc in &scenarios {
        let before = bench.checks.len();
        let span = tracing::info_span!("scenario", id = %sc.id);
        let result = {
            use tracing::Instrument;
            bench.run(sc).instrument(span).await
        };
        if let Err(e) = result {
            bench.record(
                sc,
                0,
                "error",
                "scenario failed to run",
                false,
                format!("{e:#}"),
            );
        }
        let new = &bench.checks[before..];
        eprintln!(
            "{:<45} {}/{}",
            sc.id,
            new.iter().filter(|c| c.pass).count(),
            new.len()
        );
    }
    // Totals by check kind and by scenario category.
    let mut by_kind: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut by_cat: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for c in &bench.checks {
        let e = by_kind.entry(c.kind).or_default();
        e.0 += c.pass as usize;
        e.1 += 1;
        let cats = &scenarios
            .iter()
            .find(|s| s.id == c.scenario)
            .expect("scenario")
            .categories;
        for cat in cats {
            let e = by_cat.entry(cat.clone()).or_default();
            e.0 += c.pass as usize;
            e.1 += 1;
        }
    }
    let passed = bench.checks.iter().filter(|c| c.pass).count();
    let summary = json!({
        "label": bench.args.label,
        "stt_gate": !bench.args.no_stt_gate,
        "scenarios": scenarios.len(),
        "checks": bench.checks.len(),
        "passed": passed,
        "rate": passed as f64 / bench.checks.len().max(1) as f64,
        "by_kind": by_kind.iter().map(|(k, (p, n))| (k.to_string(), json!(format!("{p}/{n}")))).collect::<serde_json::Map<_, _>>(),
        "by_category": by_cat.iter().map(|(k, (p, n))| (k.clone(), json!(format!("{p}/{n}")))).collect::<serde_json::Map<_, _>>(),
        "wall_s": started.elapsed().as_secs_f64(),
    });
    let items: Vec<Value> = bench
        .checks
        .iter()
        .map(|c| json!({"scenario": c.scenario, "session": c.session, "kind": c.kind, "what": c.what, "pass": c.pass, "detail": c.detail}))
        .collect();
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&json!({"summary": summary, "checks": items}))?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    trace.flush();
    Ok(())
}
