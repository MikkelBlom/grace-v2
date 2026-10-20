//! Grace's memory in the engine (ADR 0007, revised by ADR 0015): what she knows at the start of
//! a session, what she recalls each turn, and what she learns after each exchange.
//!
//! - **Session block**: rendered by code from the current facts, plus the commitments that are
//!   due, written into the system prompt once per session. Fixed for the session so
//!   llama-server's prompt cache stays valid; facts learned during the session are in the
//!   conversation itself anyway. Free-text session summaries are not shown: they were a second,
//!   never-corrected store of truth (a cancelled exam came back through one).
//! - **Recall** (voice path, ~40 ms): the user's words are embedded and searched; relevant facts
//!   not already in the block are attached to that user message.
//! - **Learning** (background, LLM slot 2): after each exchange the LLM proposes memory
//!   operations against related existing facts and the open commitments. Code checks them:
//!   new knowledge must quote Mikkel's own words, momentary requests are not stored, and only
//!   facts the model was shown can be changed. Every write records the turn it came from.

use std::{collections::HashSet, path::Path, sync::Arc};

use anyhow::{Context, bail};
use grace_llm::client::{ChatRequest, LlmClient};
use grace_memory::{Commitment, Fact, FactKind, Memory, Review, StyleRule};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    sync::Mutex,
};
use tracing::{info, info_span, warn};

pub const EMBEDDER: &str = "microsoft/harrier-oss-v1-270m";
pub const DIM: usize = 640;
/// Cosine distance above which a recalled fact is too unrelated to mention.
const MAX_DISTANCE: f64 = 0.55;
/// Cosine distance below which a new fact counts as a duplicate of an existing one.
const DUPLICATE_DISTANCE: f64 = 0.08;

/// Wall-clock time in Unix seconds. Injectable so the memory benchmark replays dated sessions.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

pub fn system_clock() -> Clock {
    Arc::new(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    })
}

/// Local date-time for a Unix time (Grace lives in Denmark: the machine's zone).
pub fn local_time(unix: f64) -> chrono::DateTime<chrono::Local> {
    chrono::DateTime::from_timestamp(unix as i64, 0)
        .unwrap_or_default()
        .with_timezone(&chrono::Local)
}

/// Client for `services/embed_server.py`.
pub struct Embedder {
    conn: Mutex<(
        tokio::net::tcp::OwnedWriteHalf,
        BufReader<tokio::net::tcp::OwnedReadHalf>,
    )>,
}

impl Embedder {
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        let stream = TcpStream::connect(addr).await.with_context(|| {
            format!("embed server not reachable at {addr}; start services/embed_server.py")
        })?;
        stream.set_nodelay(true)?;
        let (r, w) = stream.into_split();
        Ok(Self {
            conn: Mutex::new((w, BufReader::new(r))),
        })
    }

    pub async fn embed(&self, kind: &str, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut conn = self.conn.lock().await;
        let msg = json!({ "id": 1, "kind": kind, "texts": texts }).to_string() + "\n";
        conn.0.write_all(msg.as_bytes()).await?;
        let mut line = String::new();
        conn.1.read_line(&mut line).await?;
        let v: Value = serde_json::from_str(&line).context("bad embed reply")?;
        if let Some(e) = v["error"].as_str() {
            bail!("embed: {e}");
        }
        Ok(v["vectors"]
            .as_array()
            .context("no vectors")?
            .iter()
            .map(|row| {
                row.as_array().map_or_else(Vec::new, |r| {
                    r.iter()
                        .filter_map(Value::as_f64)
                        .map(|x| x as f32)
                        .collect()
                })
            })
            .collect())
    }
}

pub struct MemoryService {
    db: Arc<std::sync::Mutex<Memory>>,
    embedder: Embedder,
    llm: LlmClient,
    session: i64,
    block: String,
    block_ids: HashSet<i64>,
    clock: Clock,
}

impl MemoryService {
    /// Opens memory for a new live session.
    pub async fn open(path: &Path, embed_addr: &str, llm: LlmClient) -> anyhow::Result<Arc<Self>> {
        // Not reconciled at start: grace-memory-bench measured 214/246 with the pass vs 216-217
        // without, and Grace's own words crept back in (self_echo 28 -> 23). Kept for later work.
        Self::open_at(path, embed_addr, llm, system_clock()).await
    }

    /// Opens memory for a new session at `clock`'s time.
    pub async fn open_at(
        path: &Path,
        embed_addr: &str,
        llm: LlmClient,
        clock: Clock,
    ) -> anyhow::Result<Arc<Self>> {
        let embedder = Embedder::connect(embed_addr).await?;
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        let db = Memory::open(path, EMBEDDER, DIM)?;
        let now = clock();
        let session = db.start_session(now)?;
        let facts = db.current_facts(60)?;
        let due: Vec<Commitment> = db
            .commitments()?
            .into_iter()
            .filter(|c| c.open && c.due_at.is_none_or(|d| d <= now))
            .collect();
        let block = session_block(&facts, &due);
        let block_ids = facts.iter().map(|f| f.id).collect();
        info!(
            facts = facts.len(),
            due_commitments = due.len(),
            session,
            "memory loaded"
        );
        Ok(Arc::new(Self {
            db: Arc::new(std::sync::Mutex::new(db)),
            embedder,
            llm,
            session,
            block,
            block_ids,
            clock,
        }))
    }

    fn now(&self) -> f64 {
        (self.clock)()
    }

    /// Text for the system prompt: what Grace remembers coming into this session.
    pub fn session_block(&self) -> &str {
        &self.block
    }

    /// The memory part of the system message: the session block (fixed) and the style rules
    /// Mikkel taught her (live: a rule learned on one turn applies from the next). Rules change
    /// rarely, so the cached prompt prefix survives all other turns (ADR 0017).
    pub fn prompt_block(&self) -> String {
        let rules = match self.db().style_rules() {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "style: reading rules failed");
                Vec::new()
            }
        };
        let rules = crate::style::render_rules(&rules);
        if rules.is_empty() {
            self.block.clone()
        } else {
            format!(
                "{}

{rules}",
                self.block
            )
        }
    }

    /// Current style rules, oldest first.
    pub fn style_rules(&self) -> anyhow::Result<Vec<StyleRule>> {
        self.db().style_rules()
    }

    /// The review queue (`status` None = all).
    pub fn reviews(&self, status: Option<&str>) -> anyhow::Result<Vec<Review>> {
        self.db().reviews(status)
    }

    /// Current facts, for inspection (memory lab, debugging).
    pub fn facts(&self) -> anyhow::Result<Vec<Fact>> {
        self.db().current_facts(200)
    }

    /// Facts that were invalidated or superseded, newest first.
    pub fn closed_facts(&self) -> anyhow::Result<Vec<Fact>> {
        self.db().recently_closed(0.0, 1000)
    }

    /// All commitments, open first.
    pub fn commitments(&self) -> anyhow::Result<Vec<Commitment>> {
        self.db().commitments()
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Memory> {
        self.db.lock().expect("memory poisoned")
    }

    pub fn add_turn(&self, role: &str, text: &str) -> anyhow::Result<i64> {
        self.db().add_turn(self.session, role, text, self.now())
    }

    /// Facts relevant to `text` that aren't already in the session block.
    pub async fn recall(&self, text: &str) -> anyhow::Result<Vec<Fact>> {
        let v = self.embedder.embed("query", &[text]).await?.remove(0);
        let hits = self.db().search(text, &v, 5)?;
        Ok(hits
            .into_iter()
            .filter(|h| h.distance.is_some_and(|d| d <= MAX_DISTANCE))
            .filter(|h| !self.block_ids.contains(&h.fact.id))
            .map(|h| h.fact)
            .collect())
    }

    /// After an exchange: ask the LLM what to remember, and apply it. Runs in the background.
    pub async fn learn(self: Arc<Self>, user_turn: i64, user: String, reply: String) {
        let span = info_span!("memory.learn", user_turn);
        // Style first: it is one short call, and the next turn should already follow a new rule.
        if let Err(e) = self.learn_style(user_turn, &user, &reply).await {
            warn!(parent: &span, error = %e, "style learning failed");
        }
        if let Err(e) = self.learn_inner(user_turn, &user, &reply).await {
            warn!(parent: &span, error = %e, "memory learning failed");
        }
    }

    async fn learn_style(&self, user_turn: i64, user: &str, reply: &str) -> anyhow::Result<()> {
        let (previous, context) = {
            let db = self.db();
            (
                db.reply_before(self.session, user_turn)?,
                db.turns_before(self.session, user_turn, crate::style::REVIEW_CONTEXT)?,
            )
        };
        let ex = crate::style::Exchange {
            session: Some(self.session),
            user_turn: Some(user_turn),
            user,
            reply,
            previous: previous.as_deref(),
            context: &context,
        };
        crate::style::learn(&self.llm, &self.db, &ex, self.now()).await?;
        Ok(())
    }

    async fn learn_inner(&self, user_turn: i64, user: &str, reply: &str) -> anyhow::Result<()> {
        let v = self.embedder.embed("query", &[user]).await?.remove(0);
        let related = self.db().search(user, &v, 6)?;
        let related_text = if related.is_empty() {
            "(none)".to_owned()
        } else {
            related
                .iter()
                .map(|h| format!("[{}] {}", h.fact.id, h.fact.text))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let open: Vec<Commitment> = self
            .db()
            .commitments()?
            .into_iter()
            .filter(|c| c.open)
            .collect();
        let open_text = if open.is_empty() {
            "(none)".to_owned()
        } else {
            open.iter()
                .map(|c| match c.due_at {
                    Some(d) => format!(
                        "[{}] {} (from {})",
                        c.id,
                        c.text,
                        local_time(d).format("%-d %B %Y")
                    ),
                    None => format!("[{}] {}", c.id, c.text),
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let prompt = format!(
            "{}\n\nMikkel just said (spoken Danish, machine-transcribed):\n{user}\n\n\
             Grace answered (context only: never store anything that appears only here):\n{reply}\n\n\
             Existing related memories:\n{related_text}\n\n\
             Open commitments (things Grace should bring up):\n{open_text}",
            calendar(self.now())
        );
        let mut req = ChatRequest::new(vec![
            json!({ "role": "system", "content": EXTRACT_SYSTEM }),
            json!({ "role": "user", "content": prompt }),
        ]);
        req.max_tokens = 500;
        req.temperature = 0.2;
        let out = self.llm.complete_json(&req, &extract_schema()).await?;
        let ops = out["ops"].as_array().cloned().unwrap_or_default();
        let known: HashSet<i64> = related.iter().map(|h| h.fact.id).collect();
        let open_ids: HashSet<i64> = open.iter().map(|c| c.id).collect();
        for op in ops {
            self.apply(&op, user_turn, user, &known, &open_ids).await?;
        }
        Ok(())
    }

    /// End-of-session pass: with the whole conversation in view, fix what per-turn learning got
    /// wrong (a correction later in the session, a detail split over turns, something missed).
    /// Experimental: off live (see `open`); `grace-memory-bench --reconcile` measures it.
    pub async fn reconcile_session(&self, session: i64) -> anyhow::Result<()> {
        let turns = self.db().session_turns(session)?;
        let Some(last_user) = turns
            .iter()
            .rev()
            .find(|(_, r, _)| r == "user")
            .map(|t| t.0)
        else {
            return Ok(());
        };
        let said: Vec<&str> = turns
            .iter()
            .filter(|(_, r, _)| r == "user")
            .map(|(_, _, t)| t.as_str())
            .collect();
        let transcript = turns
            .iter()
            .map(|(_, r, t)| format!("{}: {t}", if r == "user" { "Mikkel" } else { "Grace" }))
            .collect::<Vec<_>>()
            .join("\n");
        let facts = self.db().current_facts(60)?;
        let open: Vec<Commitment> = self
            .db()
            .commitments()?
            .into_iter()
            .filter(|c| c.open)
            .collect();
        let list = |v: Vec<String>| {
            if v.is_empty() {
                "(none)".to_owned()
            } else {
                v.join("\n")
            }
        };
        let prompt = format!(
            "{}\n\nThe whole conversation (Mikkel's words are machine-transcribed Danish; Grace's are context only):\n{transcript}\n\n\
             Current memories:\n{}\n\nOpen commitments:\n{}",
            calendar(self.now()),
            list(
                facts
                    .iter()
                    .map(|f| format!("[{}] {}", f.id, f.text))
                    .collect()
            ),
            list(
                open.iter()
                    .map(|c| format!("[{}] {}", c.id, c.text))
                    .collect()
            ),
        );
        let mut req = ChatRequest::new(vec![
            json!({ "role": "system", "content": format!("{EXTRACT_SYSTEM}\n\n{RECONCILE_EXTRA}") }),
            json!({ "role": "user", "content": prompt }),
        ]);
        req.max_tokens = 700;
        req.temperature = 0.2;
        let out = self.llm.complete_json(&req, &extract_schema()).await?;
        let known: HashSet<i64> = facts.iter().map(|f| f.id).collect();
        let open_ids: HashSet<i64> = open.iter().map(|c| c.id).collect();
        let said = said.join("\n");
        for op in out["ops"].as_array().cloned().unwrap_or_default() {
            self.apply(&op, last_user, &said, &known, &open_ids).await?;
        }
        self.db().set_reconciled(session, self.now())?;
        info!(session, "memory: session reconciled");
        Ok(())
    }

    /// Reconciles earlier sessions that ended without it (Grace was stopped). Background work.
    pub async fn reconcile_pending(self: Arc<Self>) {
        let pending = match self.db().unreconciled_sessions(self.session) {
            Ok(p) => p,
            Err(e) => return warn!(error = %e, "memory: listing sessions failed"),
        };
        for session in pending {
            if let Err(e) = self.reconcile_session(session).await {
                warn!(error = %e, session, "memory: reconciliation failed");
            }
        }
    }

    /// This session's id (the benchmark reconciles it when a scripted session ends).
    pub fn session(&self) -> i64 {
        self.session
    }

    async fn apply(
        &self,
        op: &Value,
        user_turn: i64,
        user: &str,
        known: &HashSet<i64>,
        open: &HashSet<i64>,
    ) -> anyhow::Result<()> {
        let text = op["text"].as_str().unwrap_or("").trim().to_owned();
        let evidence = op["evidence"].as_str().unwrap_or("");
        let kind = match op["kind"].as_str() {
            Some("preference") => FactKind::Preference,
            Some("grace") => FactKind::Grace,
            Some("environment") => FactKind::Environment,
            _ => FactKind::User,
        };
        let confidence = op["confidence"].as_f64().unwrap_or(0.7);
        let id = op["id"].as_i64().unwrap_or(0);
        let name = op["op"].as_str().unwrap_or("");
        let now = self.now();
        // Provenance checked in code, not only asked for in the prompt: new knowledge must quote
        // Mikkel's own words (the baseline stored a music taste only Grace had guessed).
        if matches!(name, "add" | "update" | "commit") && !supported_by(evidence, user) {
            info!(op = name, text = %text, evidence, "memory: not in Mikkel's words, skipped");
            return Ok(());
        }
        if matches!(name, "add" | "update") && op["lasting"].as_bool() == Some(false) {
            info!(op = name, text = %text, "memory: only for now, not stored");
            return Ok(());
        }
        // A model sometimes updates an id it invented; the knowledge itself is still Mikkel's
        // (evidence checked above), so keep it as a new memory rather than drop it.
        let name = if name == "update" && !known.contains(&id) {
            info!(id, text = %text, "memory: update of an unknown id kept as add");
            "add"
        } else {
            name
        };
        match name {
            "add" if !text.is_empty() => {
                let e = self.embedder.embed("document", &[&text]).await?.remove(0);
                // Near-duplicates of a current fact are dropped (a live run stored
                // "Sofie lives in Copenhagen" twice).
                // Embeddings barely see numbers and names ("15 October" vs "16 October" scored
                // as duplicates and a correction was lost), so the specifics must match too.
                let dup = self
                    .db()
                    .search(&text, &e, 1)?
                    .into_iter()
                    .next()
                    .filter(|h| h.distance.is_some_and(|d| d < DUPLICATE_DISTANCE))
                    .filter(|h| specifics(&h.fact.text) == specifics(&text));
                if let Some(h) = dup {
                    info!(existing = h.fact.id, text = %text, "memory: duplicate skipped");
                    return Ok(());
                }
                let new = self
                    .db()
                    .add_fact(&text, kind, confidence, Some(user_turn), &e, now)?;
                info!(fact = new, text = %text, "memory: learned");
            }
            // Only facts the model was shown may be changed: no editing by guessed id.
            "update" if known.contains(&id) && !text.is_empty() => {
                let e = self.embedder.embed("document", &[&text]).await?.remove(0);
                let new =
                    self.db()
                        .supersede(id, &text, kind, confidence, Some(user_turn), &e, now)?;
                info!(old = id, fact = new, text = %text, "memory: corrected");
            }
            "invalidate" if known.contains(&id) => {
                self.db().invalidate(id, now)?;
                info!(fact = id, "memory: no longer true");
            }
            "commit" if !text.is_empty() => {
                let due = op["due"].as_str().and_then(parse_day);
                let new = self.db().add_commitment(&text, due, Some(user_turn), now)?;
                info!(commitment = new, text = %text, "memory: will bring up");
            }
            "done" if open.contains(&id) => {
                self.db().close_commitment(id, true, now)?;
                info!(commitment = id, "memory: commitment done");
            }
            "cancel" if open.contains(&id) => {
                self.db().close_commitment(id, false, now)?;
                info!(commitment = id, "memory: commitment cancelled");
            }
            other => info!(op = other, id, "memory: ignored operation"),
        }
        Ok(())
    }
}

/// What Grace knows coming into a session, rendered by code from the current facts and the
/// commitments that are due (ADR 0015).
fn session_block(facts: &[Fact], due: &[Commitment]) -> String {
    if facts.is_empty() && due.is_empty() {
        return "You don't remember anything about Mikkel yet: this is your first conversation with memory.".into();
    }
    let mut out = String::new();
    if !due.is_empty() {
        // First, and phrased as an instruction: in the baseline a reminder listed among the
        // facts was raised in only 6 of 10 sessions.
        out.push_str(
            "Mikkel asked you to bring these up. Do it in your first reply of this conversation, \
             whatever he opens with:\n",
        );
        for c in due {
            out.push_str(&format!("- {}\n", c.text));
        }
        out.push('\n');
    }
    if !facts.is_empty() {
        out.push_str(
            "What you know about Mikkel (current; he can correct you; anything not listed here \
             you don't know):\n",
        );
        for f in facts {
            out.push_str(&format!("- {}\n", f.text));
        }
    }
    out
}

/// Today plus the next two weeks with weekdays, so relative dates ("på mandag", "næste fredag")
/// are looked up rather than computed by the model (baseline: 58 of 80 temporal checks).
pub fn calendar(now: f64) -> String {
    let today = local_time(now).date_naive();
    let days: Vec<String> = (1..=14)
        .map(|i| {
            (today + chrono::Days::new(i))
                .format("%A %-d %B")
                .to_string()
        })
        .collect();
    format!(
        "Today is {}. The next days: {}.",
        today.format("%A %-d %B %Y"),
        days.join(", ")
    )
}

/// Start of a `YYYY-MM-DD` day in local time, as Unix seconds.
fn parse_day(s: &str) -> Option<f64> {
    let d = chrono::NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()?;
    let t = d
        .and_hms_opt(0, 0, 0)?
        .and_local_timezone(chrono::Local)
        .earliest()?;
    Some(t.timestamp() as f64)
}

const NUMBER_WORDS: &[&str] = &[
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "first",
    "second",
    "third",
    "half",
];

const MONTHS_AND_DAYS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

/// Capitalised words that are not names.
const NOT_NAMES: &[&str] = &[
    "Mikkel", "Grace", "He", "She", "His", "Her", "They", "Their", "The", "A", "An", "It", "On",
    "In", "At",
];

/// The details an embedding glosses over: numbers (digits or words), dates, and names
/// (capitalised words). Two
/// facts with different specifics are never duplicates.
fn specifics(text: &str) -> std::collections::BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && *w != "s")
        .filter(|w| {
            let lower = w.to_lowercase();
            w.chars().any(|c| c.is_ascii_digit())
                || MONTHS_AND_DAYS.contains(&lower.as_str())
                || NUMBER_WORDS.contains(&lower.as_str())
                || (w.chars().next().is_some_and(char::is_uppercase) && !NOT_NAMES.contains(w))
        })
        .map(str::to_lowercase)
        .collect()
}

/// Whether `evidence` is (nearly) a quote of `said`: at least 80 % of its words occur there.
pub(crate) fn supported_by(evidence: &str, said: &str) -> bool {
    let words = |s: &str| -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect()
    };
    let (ev, said) = (words(evidence), words(said));
    if ev.is_empty() {
        return false;
    }
    let found = ev.iter().filter(|w| said.contains(w)).count();
    found * 10 >= ev.len() * 8
}

/// Appends recalled facts to the user's words for this turn only.
pub fn with_recall(user: &str, recalled: &[Fact]) -> String {
    if recalled.is_empty() {
        return user.to_owned();
    }
    let lines: Vec<String> = recalled.iter().map(|f| format!("- {}", f.text)).collect();
    format!(
        "{user}\n\n[Possibly relevant things you remember:\n{}]",
        lines.join("\n")
    )
}

const EXTRACT_SYSTEM: &str = "You maintain Grace's long-term memory about Mikkel, the person she talks to. From the latest exchange, decide what is worth remembering beyond today.
Remember: facts about Mikkel and the people in his life, his projects, plans and dates, lasting preferences about what Grace should do, and corrections of what Grace believed. Not how Grace talks (reply length, tone, humour, questions, wording, speaking pace): that feedback is kept separately as style rules, so never store it here.
Only Mikkel's own words are a source. Grace's answer is context: she may be repeating something outdated or guessing, so never store anything because Grace said it. For every add, update and commit, 'evidence' must quote the words of Mikkel that state it, copied exactly from what he said; if you can't quote him, don't store it.
Speech recognition makes mistakes: if his words look garbled, nonsensical or like a misheard phrase, store nothing from them.
Do not remember: small talk, greetings, questions he asked, tests of the system, vague moods, or anything already covered by an existing memory.
'lasting' is true only for things that stay true beyond today. Requests for this moment or this conversation ('be quiet', 'speak slower', 'say it again', 'not today', 'hold din kæft') are lasting=false; a lasting wish sounds like 'always', 'from now on', 'in general'.
Write each memory in English, third person ('Mikkel ...'), short, one claim per memory, numbers as digits ('turns 95', '52 guests'). Turn relative dates into absolute dates by looking them up in the calendar above (never compute weekdays yourself), and write dates in words: '14 December 2026', never '2026-12-14'. Never add details he didn't say: no invented dates, places or reasons.
Kinds: 'user' = Mikkel's life, people, plans, projects and circumstances; 'preference' = what Mikkel wants Grace to do or how she should handle things (never how she talks); 'grace' = about Grace herself; 'environment' = Mikkel's computer, devices and setup.
Operations on memories: 'add' (id 0) for a new memory. 'update' with an existing id when the exchange changes that memory (a new value, a corrected detail). 'invalidate' with an existing id when it is simply no longer true: cancelled, finished, sold, ended (do not add a 'was cancelled' memory). Never leave two memories that contradict each other.
Operations on commitments (things Grace should bring up later): 'commit' only when Mikkel explicitly asks Grace to remind him, ask him, or check on something later (never on your own initiative); 'text' says what Grace should do ('Ask Mikkel for the details of his 16 November assignment'); 'due' (only this field uses YYYY-MM-DD) is the first day to bring it up, or '' for the next conversation. 'done' with an open commitment's id when this exchange fulfils it: he gave the details, did the thing, or acknowledged the reminder. 'cancel' with an open commitment's id when he no longer wants it. A commitment never replaces facts: also add the facts he stated (a reminder to renew a passport that expires in February is a commitment plus the fact about the expiry).
Confidence: 0.9 when Mikkel stated it plainly, 0.6 when inferred.
Most exchanges contain nothing worth remembering: then return an empty list.";

const RECONCILE_EXTRA: &str = "This time you see a whole conversation at once, after per-exchange learning already ran. Make the memories match what Mikkel said by the END of the conversation: a later statement overrides an earlier one ('on Thursday ... no wait, Friday' leaves only Friday). Fix memories that are wrong, split, or missing; invalidate memories the conversation shows are no longer true; mark commitments done or cancelled when the conversation settled them. Evidence may quote any of Mikkel's lines. If the memories already match, return an empty list.";

const SUMMARY_SYSTEM: &str = "Summarise this conversation between Mikkel and Grace (his voice assistant) for \
Grace's memory in two to four short English sentences: what was discussed, what was decided, and anything \
left open. No preamble.";

impl MemoryService {
    /// Writes a summary for each earlier session that has none. Summaries are no longer shown
    /// to Grace (ADR 0015); they are kept for future episodic search.
    pub async fn summarize_old_sessions(self: Arc<Self>) {
        let pending = match self.db().unsummarized_sessions(self.session) {
            Ok(p) => p,
            Err(e) => return warn!(error = %e, "memory: listing sessions failed"),
        };
        for (session, turns) in pending {
            let transcript = turns
                .iter()
                .map(|(r, t)| format!("{}: {t}", if r == "user" { "Mikkel" } else { "Grace" }))
                .collect::<Vec<_>>()
                .join("\n");
            let mut req = ChatRequest::new(vec![
                json!({ "role": "system", "content": SUMMARY_SYSTEM }),
                json!({ "role": "user", "content": transcript }),
            ]);
            req.max_tokens = 200;
            req.temperature = 0.2;
            let schema = json!({"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]});
            match self.llm.complete_json(&req, &schema).await {
                Ok(v) => {
                    let summary = v["summary"].as_str().unwrap_or("").trim().to_owned();
                    if let Err(e) = self.db().set_summary(session, &summary, self.now()) {
                        warn!(error = %e, session, "memory: storing summary failed");
                    } else {
                        info!(session, summary = %summary, "memory: session summarised");
                    }
                }
                Err(e) => warn!(error = %e, session, "memory: summary failed"),
            }
        }
    }
}

fn extract_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": { "type": "string", "enum": ["add", "update", "invalidate", "commit", "done", "cancel"] },
                        "id": { "type": "integer" },
                        "evidence": { "type": "string" },
                        "text": { "type": "string" },
                        "kind": { "type": "string", "enum": ["user", "preference", "grace", "environment"] },
                        "lasting": { "type": "boolean" },
                        "due": { "type": "string" },
                        "confidence": { "type": "number" }
                    },
                    "required": ["op", "id", "evidence", "text", "kind", "lasting", "due", "confidence"]
                }
            }
        },
        "required": ["ops"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_is_appended_only_when_there_is_something() {
        assert_eq!(with_recall("hej", &[]), "hej");
        let f = Fact {
            id: 1,
            text: "Mikkel's sister is called Sofie.".into(),
            kind: FactKind::User,
            confidence: 0.9,
            source_turn: None,
            recorded_at: 0.0,
        };
        let s = with_recall("hvem er min søster", &[f]);
        assert!(s.starts_with("hvem er min søster\n\n[Possibly relevant"));
        assert!(s.contains("Sofie"));
    }

    #[test]
    fn empty_memory_says_so() {
        assert!(session_block(&[], &[]).contains("first conversation"));
    }

    #[test]
    fn due_commitments_come_first() {
        let c = Commitment {
            id: 1,
            text: "Ask Mikkel about the hand-in".into(),
            due_at: None,
            open: true,
            source_turn: None,
        };
        let b = session_block(&[], &[c]);
        assert!(b.starts_with("Mikkel asked you to bring these up"));
        assert!(b.contains("hand-in"));
    }

    #[test]
    fn evidence_must_quote_mikkel() {
        let said = "min søster hedder Sofie og hun bor i Aarhus";
        assert!(supported_by("hun bor i Aarhus", said));
        assert!(supported_by("Min søster hedder Sofie,", said));
        assert!(
            !supported_by("jeg kan godt lide jazz", said),
            "Grace's guess, not his words"
        );
        assert!(!supported_by("", said));
    }

    #[test]
    fn different_dates_or_names_are_not_duplicates() {
        let a = "Mikkel has a dentist appointment on 15 October 2026 at 10:00.";
        let b = "Mikkel has a dentist appointment on 16 October 2026 at 10:00.";
        assert_ne!(specifics(a), specifics(b));
        assert_ne!(
            specifics("Mikkel's sister Ida lives in Odense."),
            specifics("Mikkel's sister Sofie lives in Odense.")
        );
        assert_ne!(
            specifics("Mikkel's grandfather turns ninety on 21 November 2026."),
            specifics("Mikkel's grandfather turns ninety-five on 21 November 2026.")
        );
        assert_eq!(
            specifics("Mikkel's sister Sofie lives in Copenhagen."),
            specifics("Sofie, Mikkel's sister, lives in Copenhagen.")
        );
    }

    #[test]
    fn calendar_lists_weekdays_ahead() {
        // 12 October 2026 is a Monday.
        let t = parse_day("2026-10-12").expect("date") + 10.0 * 3600.0;
        let c = calendar(t);
        assert!(c.starts_with("Today is Monday 12 October 2026."), "{c}");
        assert!(c.contains("Monday 19 October"));
    }
}
