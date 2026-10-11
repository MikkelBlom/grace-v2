//! Grace's memory in the engine (ADR 0007): what she knows at the start of a session, what she
//! recalls each turn, and what she learns after each exchange.
//!
//! - **Session block**: current facts and recent session summaries, written into the system
//!   prompt once per session. Fixed for the session so llama-server's prompt cache stays valid;
//!   facts learned during the session are in the conversation itself anyway.
//! - **Recall** (voice path, ~40 ms): the user's words are embedded and searched; relevant facts
//!   not already in the block are attached to that user message.
//! - **Learning** (background, LLM slot 2): after each exchange the LLM proposes memory
//!   operations (add / update / invalidate) against related existing facts; every write records
//!   the turn it came from.
//! - **Summaries**: sessions without a summary (Grace was stopped) are summarised at the next
//!   start, in the background.

use std::{collections::HashSet, path::Path, sync::Arc};

use anyhow::{Context, bail};
use grace_llm::client::{ChatRequest, LlmClient};
use grace_memory::{Fact, FactKind, Memory};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    sync::Mutex,
};
use tracing::{info, info_span, warn};

const EMBEDDER: &str = "microsoft/harrier-oss-v1-270m";
const DIM: usize = 640;
/// Cosine distance above which a recalled fact is too unrelated to mention.
const MAX_DISTANCE: f64 = 0.55;
/// Cosine distance below which a new fact counts as a duplicate of an existing one.
const DUPLICATE_DISTANCE: f64 = 0.08;

fn now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
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
}

impl MemoryService {
    pub async fn open(path: &Path, embed_addr: &str, llm: LlmClient) -> anyhow::Result<Arc<Self>> {
        let embedder = Embedder::connect(embed_addr).await?;
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        let db = Memory::open(path, EMBEDDER, DIM)?;
        let session = db.start_session(now_unix())?;
        let facts = db.current_facts(60)?;
        let summaries = db.recent_summaries(3)?;
        let block = session_block(&facts, &summaries);
        let block_ids = facts.iter().map(|f| f.id).collect();
        info!(
            facts = facts.len(),
            summaries = summaries.len(),
            session,
            "memory loaded"
        );
        let svc = Arc::new(Self {
            db: Arc::new(std::sync::Mutex::new(db)),
            embedder,
            llm,
            session,
            block,
            block_ids,
        });
        tokio::spawn(svc.clone().summarize_old_sessions());
        Ok(svc)
    }

    /// Text for the system prompt: what Grace remembers coming into this session.
    pub fn session_block(&self) -> &str {
        &self.block
    }

    /// Current facts, for inspection (memory lab, debugging).
    pub fn facts(&self) -> anyhow::Result<Vec<Fact>> {
        self.db().current_facts(200)
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Memory> {
        self.db.lock().expect("memory poisoned")
    }

    pub fn add_turn(&self, role: &str, text: &str) -> anyhow::Result<i64> {
        self.db().add_turn(self.session, role, text, now_unix())
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
        if let Err(e) = self.learn_inner(user_turn, &user, &reply).await {
            warn!(parent: &span, error = %e, "memory learning failed");
        }
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
        let today = chrono::Local::now().format("%A %-d %B %Y");
        let prompt = format!(
            "Today is {today}.\n\nLatest exchange:\nMikkel (spoken Danish, transcribed): {user}\nGrace: {reply}\n\n\
             Existing related memories:\n{related_text}"
        );
        let mut req = ChatRequest::new(vec![
            json!({ "role": "system", "content": EXTRACT_SYSTEM }),
            json!({ "role": "user", "content": prompt }),
        ]);
        req.max_tokens = 400;
        req.temperature = 0.2;
        let out = self.llm.complete_json(&req, &extract_schema()).await?;
        let ops = out["ops"].as_array().cloned().unwrap_or_default();
        let known: HashSet<i64> = related.iter().map(|h| h.fact.id).collect();
        for op in ops {
            let text = op["text"].as_str().unwrap_or("").trim().to_owned();
            let kind = match op["kind"].as_str() {
                Some("preference") => FactKind::Preference,
                Some("grace") => FactKind::Grace,
                Some("environment") => FactKind::Environment,
                _ => FactKind::User,
            };
            let confidence = op["confidence"].as_f64().unwrap_or(0.7);
            let id = op["id"].as_i64().unwrap_or(0);
            let now = now_unix();
            match op["op"].as_str() {
                Some("add") if !text.is_empty() => {
                    let e = self.embedder.embed("document", &[&text]).await?.remove(0);
                    // Near-duplicates of a current fact are dropped (a live run stored
                    // "Sofie lives in Copenhagen" twice).
                    let dup = self
                        .db()
                        .search(&text, &e, 1)?
                        .into_iter()
                        .next()
                        .filter(|h| h.distance.is_some_and(|d| d < DUPLICATE_DISTANCE));
                    if let Some(h) = dup {
                        info!(existing = h.fact.id, text = %text, "memory: duplicate skipped");
                        continue;
                    }
                    let new =
                        self.db()
                            .add_fact(&text, kind, confidence, Some(user_turn), &e, now)?;
                    info!(fact = new, text = %text, "memory: learned");
                }
                // Only facts the model was shown may be changed: no editing by guessed id.
                Some("update") if known.contains(&id) && !text.is_empty() => {
                    let e = self.embedder.embed("document", &[&text]).await?.remove(0);
                    let new = self.db().supersede(
                        id,
                        &text,
                        kind,
                        confidence,
                        Some(user_turn),
                        &e,
                        now,
                    )?;
                    info!(old = id, fact = new, text = %text, "memory: corrected");
                }
                Some("invalidate") if known.contains(&id) => {
                    self.db().invalidate(id, now)?;
                    info!(fact = id, "memory: no longer true");
                }
                other => info!(op = ?other, id, "memory: ignored operation"),
            }
        }
        Ok(())
    }

    async fn summarize_old_sessions(self: Arc<Self>) {
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
                    if let Err(e) = self.db().set_summary(session, &summary, now_unix()) {
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

fn session_block(facts: &[Fact], summaries: &[(f64, String)]) -> String {
    if facts.is_empty() && summaries.is_empty() {
        return "You don't remember anything about Mikkel yet: this is your first conversation with memory.".into();
    }
    let mut out = String::from(
        "What you remember (from earlier conversations; it may be outdated, and he can correct you):\n",
    );
    for f in facts {
        out.push_str(&format!("- {}\n", f.text));
    }
    if !summaries.is_empty() {
        out.push_str("\nRecent conversations:\n");
        for (at, s) in summaries {
            let day = chrono::DateTime::from_timestamp(*at as i64, 0)
                .map(|d| d.with_timezone(&chrono::Local).format("%-d %B").to_string())
                .unwrap_or_default();
            out.push_str(&format!("- {day}: {s}\n"));
        }
    }
    out
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
Remember: facts about Mikkel and the people in his life, his projects, plans and dates, preferences, how he wants Grace to behave, and corrections of what Grace believed.
Do not remember: small talk, greetings, questions he asked, things only Grace said, tests of the system, vague impressions, or anything already covered by an existing memory. Momentary commands are not preferences: 'be quiet', 'stop', 'say that again', 'hold din kæft' apply to that moment only; store a preference only when he states a lasting wish ('I always want short answers').
Write each memory in English, third person ('Mikkel ...'), short and self-contained. Turn relative dates ('next Tuesday') into absolute dates using today's date.
Kinds: 'user' = about Mikkel's life, people, plans, projects and circumstances; 'preference' = how Mikkel wants Grace to talk or behave; 'grace' = about Grace herself (her name, origin, abilities); 'environment' = Mikkel's computer, devices and setup.
Operations: 'add' (id 0) for a new memory. 'update' with an existing id when the exchange changes that memory (a new value, a corrected detail). 'invalidate' with an existing id when it is simply no longer true: something cancelled, finished, sold, ended. Example: existing [7] 'Mikkel has a dentist appointment on 14 October' and Mikkel says the appointment was cancelled -> invalidate 7 (do not add a 'was cancelled' memory). Never leave two memories that contradict each other.
Confidence: 0.9 when Mikkel stated it plainly, 0.6 when it is inferred.
Most exchanges contain nothing worth remembering: then return an empty list.";

const SUMMARY_SYSTEM: &str = "Summarise this conversation between Mikkel and Grace (his voice assistant) for \
Grace's memory in two to four short English sentences: what was discussed, what was decided, and anything \
left open. No preamble.";

fn extract_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": { "type": "string", "enum": ["add", "update", "invalidate"] },
                        "id": { "type": "integer" },
                        "text": { "type": "string" },
                        "kind": { "type": "string", "enum": ["user", "preference", "grace", "environment"] },
                        "confidence": { "type": "number" }
                    },
                    "required": ["op", "id", "text", "kind", "confidence"]
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
}
