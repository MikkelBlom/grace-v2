//! Grace's long-term memory (ADR 0007): one SQLite file is the source of truth.
//!
//! - **Sessions and turns**: everything said, kept verbatim (episodes), plus a summary per session.
//! - **Facts**: what Grace has learned ("Mikkel prefers short answers"). Each fact has a kind, a
//!   confidence, the turn it came from (provenance), and bitemporal validity. A fact that stops
//!   being true is closed (`valid_to`, `invalidated_at`, `superseded_by`), never deleted, so
//!   "what did we believe in June?" stays answerable and a bad write can be traced and undone.
//! - **Search**: FTS5 keyword ranking and sqlite-vec nearest neighbours, merged with reciprocal
//!   rank fusion. Only currently valid facts are returned.
//!
//! Embeddings are computed outside (the embed server); the embedder's id and dimension are
//! stored, and opening with a different embedder is refused rather than mixing vector spaces.
#![allow(unsafe_code)] // one call: registering sqlite-vec as an auto-extension

use std::{path::Path, sync::Once};

use anyhow::{Context, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

pub type Time = f64; // seconds since the Unix epoch

/// A session's turns as (role, text), oldest first.
pub type Transcript = Vec<(String, String)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactKind {
    /// About the user: people, plans, circumstances.
    User,
    /// How the user wants Grace to behave.
    Preference,
    /// About Grace herself.
    Grace,
    /// About the user's machine or environment that isn't observable live.
    Environment,
    /// Something Grace should bring up or ask about later ("remind me", "ask me about it").
    Followup,
}

impl FactKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Preference => "preference",
            Self::Grace => "grace",
            Self::Environment => "environment",
            Self::Followup => "followup",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "preference" => Self::Preference,
            "grace" => Self::Grace,
            "environment" => Self::Environment,
            "followup" => Self::Followup,
            _ => Self::User,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Fact {
    pub id: i64,
    pub text: String,
    pub kind: FactKind,
    pub confidence: f64,
    pub source_turn: Option<i64>,
    pub recorded_at: Time,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hit {
    pub fact: Fact,
    /// Reciprocal-rank-fusion score (higher is better).
    pub score: f64,
    /// Cosine distance from the query, if the vector search found it (0 = identical).
    pub distance: Option<f64>,
}

pub struct Memory {
    conn: Connection,
    dim: usize,
}

fn register_vec() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut i8,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> i32,
        >(
            sqlite_vec::sqlite3_vec_init as *const ()
        )));
    });
}

fn blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT,
    summarized_at REAL
);
CREATE TABLE IF NOT EXISTS turns(
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    at REAL NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id);
CREATE TABLE IF NOT EXISTS facts(
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_turn INTEGER REFERENCES turns(id),
    valid_from REAL NOT NULL,
    valid_to REAL,
    recorded_at REAL NOT NULL,
    invalidated_at REAL,
    superseded_by INTEGER REFERENCES facts(id)
);
CREATE INDEX IF NOT EXISTS facts_current ON facts(invalidated_at);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, content='facts', content_rowid='id', tokenize='unicode61');
";

impl Memory {
    /// Opens (or creates) the memory at `path`. `embedder` identifies the embedding model.
    pub fn open(path: &Path, embedder: &str, dim: usize) -> anyhow::Result<Self> {
        register_vec();
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::init(conn, embedder, dim)
    }

    pub fn open_in_memory(embedder: &str, dim: usize) -> anyhow::Result<Self> {
        register_vec();
        Self::init(Connection::open_in_memory()?, embedder, dim)
    }

    fn init(conn: Connection, embedder: &str, dim: usize) -> anyhow::Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        let stored: Option<String> = conn
            .query_row("SELECT value FROM meta WHERE key = 'embedder'", [], |r| {
                r.get(0)
            })
            .optional()?;
        let want = format!("{embedder}:{dim}");
        match stored {
            Some(s) if s != want => bail!(
                "memory was built with embedder {s}, not {want}; re-embed before switching models"
            ),
            Some(_) => {}
            None => {
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('embedder', ?1)",
                    [&want],
                )?;
            }
        }
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(embedding float[{dim}] distance_metric=cosine);"
        ))?;
        Ok(Self { conn, dim })
    }

    // ── sessions and turns ──────────────────────────────────────────────────────────────────

    pub fn start_session(&self, now: Time) -> anyhow::Result<i64> {
        self.conn
            .execute("INSERT INTO sessions(started_at) VALUES (?1)", [now])?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn add_turn(&self, session: i64, role: &str, text: &str, at: Time) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT INTO turns(session_id, at, role, text) VALUES (?1, ?2, ?3, ?4)",
            params![session, at, role, text],
        )?;
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?2 WHERE id = ?1",
            params![session, at],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Earlier sessions with turns but no summary yet (e.g. Grace was stopped mid-session).
    pub fn unsummarized_sessions(&self, except: i64) -> anyhow::Result<Vec<(i64, Transcript)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM sessions WHERE summary IS NULL AND id != ?1
             AND EXISTS (SELECT 1 FROM turns WHERE session_id = sessions.id) ORDER BY id",
        )?;
        let ids: Vec<i64> = stmt
            .query_map([except], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        ids.into_iter()
            .map(|id| Ok((id, self.turns(id)?)))
            .collect()
    }

    pub fn turns(&self, session: i64) -> anyhow::Result<Transcript> {
        let mut stmt = self
            .conn
            .prepare("SELECT role, text FROM turns WHERE session_id = ?1 ORDER BY id")?;
        let rows = stmt
            .query_map([session], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    pub fn set_summary(&self, session: i64, summary: &str, now: Time) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE sessions SET summary = ?2, summarized_at = ?3 WHERE id = ?1",
            params![session, summary, now],
        )?;
        Ok(())
    }

    /// The most recent session summaries, oldest first, with their start times.
    pub fn recent_summaries(&self, n: usize) -> anyhow::Result<Vec<(Time, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT started_at, summary FROM sessions WHERE summary IS NOT NULL ORDER BY id DESC LIMIT ?1",
        )?;
        let mut rows: Vec<(Time, String)> = stmt
            .query_map([n as i64], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        rows.reverse();
        Ok(rows)
    }

    // ── facts ───────────────────────────────────────────────────────────────────────────────

    fn check_dim(&self, v: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(
            v.len() == self.dim,
            "embedding has {} dims, memory expects {}",
            v.len(),
            self.dim
        );
        Ok(())
    }

    /// Records a new fact. `source_turn` is the turn it was learned from (provenance).
    pub fn add_fact(
        &mut self,
        text: &str,
        kind: FactKind,
        confidence: f64,
        source_turn: Option<i64>,
        embedding: &[f32],
        now: Time,
    ) -> anyhow::Result<i64> {
        self.check_dim(embedding)?;
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO facts(text, kind, confidence, source_turn, valid_from, recorded_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            params![
                text,
                kind.as_str(),
                confidence.clamp(0.0, 1.0),
                source_turn,
                now
            ],
        )?;
        let id = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO facts_fts(rowid, text) VALUES (?1, ?2)",
            params![id, text],
        )?;
        tx.execute(
            "INSERT INTO facts_vec(rowid, embedding) VALUES (?1, ?2)",
            params![id, blob(embedding)],
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Replaces fact `old` with a corrected version; the old one is closed, not deleted.
    #[allow(clippy::too_many_arguments)]
    pub fn supersede(
        &mut self,
        old: i64,
        text: &str,
        kind: FactKind,
        confidence: f64,
        source_turn: Option<i64>,
        embedding: &[f32],
        now: Time,
    ) -> anyhow::Result<i64> {
        let new = self.add_fact(text, kind, confidence, source_turn, embedding, now)?;
        let changed = self.conn.execute(
            "UPDATE facts SET valid_to = ?2, invalidated_at = ?2, superseded_by = ?3
             WHERE id = ?1 AND invalidated_at IS NULL",
            params![old, now, new],
        )?;
        anyhow::ensure!(changed == 1, "fact {old} is not a current fact");
        Ok(new)
    }

    /// Marks a fact as no longer true.
    pub fn invalidate(&self, id: i64, now: Time) -> anyhow::Result<()> {
        let changed = self.conn.execute(
            "UPDATE facts SET valid_to = ?2, invalidated_at = ?2 WHERE id = ?1 AND invalidated_at IS NULL",
            params![id, now],
        )?;
        anyhow::ensure!(changed == 1, "fact {id} is not a current fact");
        Ok(())
    }

    fn row_to_fact(r: &rusqlite::Row<'_>) -> rusqlite::Result<Fact> {
        Ok(Fact {
            id: r.get(0)?,
            text: r.get(1)?,
            kind: FactKind::parse(&r.get::<_, String>(2)?),
            confidence: r.get(3)?,
            source_turn: r.get(4)?,
            recorded_at: r.get(5)?,
        })
    }

    const FACT_COLS: &'static str = "id, text, kind, confidence, source_turn, recorded_at";

    /// Currently valid facts, most recently recorded first.
    pub fn current_facts(&self, limit: usize) -> anyhow::Result<Vec<Fact>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {} FROM facts WHERE invalidated_at IS NULL ORDER BY recorded_at DESC, id DESC LIMIT ?1",
            Self::FACT_COLS
        ))?;
        let rows = stmt
            .query_map([limit as i64], Self::row_to_fact)?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    /// Facts closed since `since` (invalidated or superseded), newest first: what Grace must not
    /// bring up as current even if an old conversation summary still mentions it.
    pub fn recently_closed(&self, since: Time, limit: usize) -> anyhow::Result<Vec<Fact>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {} FROM facts WHERE invalidated_at IS NOT NULL AND invalidated_at >= ?1
             ORDER BY invalidated_at DESC LIMIT ?2",
            Self::FACT_COLS
        ))?;
        let rows = stmt
            .query_map(params![since, limit as i64], Self::row_to_fact)?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    }

    pub fn fact(&self, id: i64) -> anyhow::Result<Option<Fact>> {
        Ok(self
            .conn
            .query_row(
                &format!("SELECT {} FROM facts WHERE id = ?1", Self::FACT_COLS),
                [id],
                Self::row_to_fact,
            )
            .optional()?)
    }

    /// Hybrid search over current facts: keyword (FTS5/BM25) and vector (cosine) rankings,
    /// merged with reciprocal rank fusion (k = 60).
    pub fn search(&self, query: &str, embedding: &[f32], k: usize) -> anyhow::Result<Vec<Hit>> {
        self.check_dim(embedding)?;
        let pool = (k * 4).max(20);
        let mut scores: std::collections::HashMap<i64, (f64, Option<f64>)> = Default::default();

        // Vector neighbours (over-fetch: invalidated facts are filtered afterwards).
        let mut stmt = self.conn.prepare(
            "SELECT v.rowid, v.distance FROM facts_vec v JOIN facts f ON f.id = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2 AND f.invalidated_at IS NULL ORDER BY v.distance",
        )?;
        let vec_hits: Vec<(i64, f64)> = stmt
            .query_map(params![blob(embedding), pool as i64], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?
            .collect::<Result<_, _>>()?;
        for (rank, (id, dist)) in vec_hits.into_iter().enumerate() {
            let e = scores.entry(id).or_insert((0.0, None));
            e.0 += 1.0 / (60.0 + rank as f64 + 1.0);
            e.1 = Some(dist);
        }

        // Keywords: any query word, BM25-ranked.
        let words: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() >= 3)
            .map(|w| format!("\"{}\"", w.to_lowercase()))
            .collect();
        if !words.is_empty() {
            let mut stmt = self.conn.prepare(
                "SELECT facts_fts.rowid FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid
                 WHERE facts_fts MATCH ?1 AND f.invalidated_at IS NULL ORDER BY bm25(facts_fts) LIMIT ?2",
            )?;
            let ids: Vec<i64> = stmt
                .query_map(params![words.join(" OR "), pool as i64], |r| r.get(0))?
                .collect::<Result<_, _>>()?;
            for (rank, id) in ids.into_iter().enumerate() {
                scores.entry(id).or_insert((0.0, None)).0 += 1.0 / (60.0 + rank as f64 + 1.0);
            }
        }

        let mut ranked: Vec<(i64, f64, Option<f64>)> =
            scores.into_iter().map(|(id, (s, d))| (id, s, d)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(k)
            .filter_map(|(id, score, distance)| match self.fact(id) {
                Ok(Some(fact)) => Some(Ok(Hit {
                    fact,
                    score,
                    distance,
                })),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic "embedding": a one-hot-ish vector per topic word.
    fn emb(topics: &[usize]) -> Vec<f32> {
        let mut v = [0.01f32; 8];
        for &t in topics {
            v[t] = 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn mem() -> Memory {
        Memory::open_in_memory("test-embedder", 8).unwrap()
    }

    #[test]
    fn stores_turns_and_summaries_per_session() {
        let m = mem();
        let s1 = m.start_session(100.0).unwrap();
        m.add_turn(s1, "user", "Hej Grace", 101.0).unwrap();
        m.add_turn(s1, "assistant", "Hello!", 102.0).unwrap();
        let s2 = m.start_session(200.0).unwrap();
        let pending = m.unsummarized_sessions(s2).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].1[0], ("user".into(), "Hej Grace".into()));
        m.set_summary(s1, "Said hello.", 300.0).unwrap();
        assert!(m.unsummarized_sessions(s2).unwrap().is_empty());
        assert_eq!(
            m.recent_summaries(5).unwrap(),
            vec![(100.0, "Said hello.".into())]
        );
    }

    #[test]
    fn vector_and_keyword_search_find_current_facts() {
        let mut m = mem();
        let coffee = m
            .add_fact(
                "Mikkel drinks his coffee black.",
                FactKind::Preference,
                0.9,
                None,
                &emb(&[0]),
                1.0,
            )
            .unwrap();
        m.add_fact(
            "Mikkel runs a Coding Pirates team.",
            FactKind::User,
            0.9,
            None,
            &emb(&[1]),
            2.0,
        )
        .unwrap();
        m.add_fact(
            "Mikkel's speakers are Logitech Z337.",
            FactKind::Environment,
            0.9,
            None,
            &emb(&[2]),
            3.0,
        )
        .unwrap();

        let hits = m
            .search("hvordan drikker jeg kaffe", &emb(&[0]), 2)
            .unwrap();
        assert_eq!(hits[0].fact.id, coffee, "vector match wins");
        assert!(hits[0].distance.unwrap() < 0.01);

        // Keyword-only match (vector points elsewhere) still surfaces.
        let hits = m.search("Logitech", &emb(&[5]), 3).unwrap();
        assert!(hits.iter().any(|h| h.fact.text.contains("Logitech")));
    }

    #[test]
    fn supersede_closes_the_old_fact_and_keeps_history() {
        let mut m = mem();
        let s = m.start_session(0.0).unwrap();
        let t = m
            .add_turn(s, "user", "jeg drikker faktisk mælk i kaffen nu", 5.0)
            .unwrap();
        let old = m
            .add_fact(
                "Mikkel drinks his coffee black.",
                FactKind::Preference,
                0.9,
                None,
                &emb(&[0]),
                1.0,
            )
            .unwrap();
        let new = m
            .supersede(
                old,
                "Mikkel takes milk in his coffee.",
                FactKind::Preference,
                0.9,
                Some(t),
                &emb(&[0]),
                6.0,
            )
            .unwrap();
        let current = m.current_facts(10).unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].id, new);
        assert_eq!(current[0].source_turn, Some(t));
        let hits = m.search("kaffe", &emb(&[0]), 5).unwrap();
        assert!(
            hits.iter().all(|h| h.fact.id != old),
            "invalidated facts are not returned"
        );
        assert!(
            m.fact(old).unwrap().is_some(),
            "but still exist for history"
        );
        assert!(
            m.supersede(old, "x", FactKind::User, 0.5, None, &emb(&[0]), 7.0)
                .is_err()
        );
    }

    #[test]
    fn recently_closed_lists_invalidated_and_superseded() {
        let mut m = mem();
        let a = m
            .add_fact(
                "Mikkel has an exam on 14 December.",
                FactKind::User,
                0.9,
                None,
                &emb(&[0]),
                1.0,
            )
            .unwrap();
        let b = m
            .add_fact(
                "Sofie lives in Aarhus.",
                FactKind::User,
                0.9,
                None,
                &emb(&[1]),
                1.0,
            )
            .unwrap();
        m.invalidate(a, 10.0).unwrap();
        m.supersede(
            b,
            "Sofie lives in Copenhagen.",
            FactKind::User,
            0.9,
            None,
            &emb(&[1]),
            11.0,
        )
        .unwrap();
        let closed: Vec<String> = m
            .recently_closed(5.0, 10)
            .unwrap()
            .into_iter()
            .map(|f| f.text)
            .collect();
        assert_eq!(
            closed,
            vec![
                "Sofie lives in Aarhus.",
                "Mikkel has an exam on 14 December."
            ]
        );
        assert!(m.recently_closed(20.0, 10).unwrap().is_empty());
        assert_eq!(
            FactKind::parse(FactKind::Followup.as_str()),
            FactKind::Followup
        );
    }

    #[test]
    fn invalidate_hides_fact() {
        let mut m = mem();
        let id = m
            .add_fact(
                "Mikkel has a dentist appointment on Tuesday.",
                FactKind::User,
                0.8,
                None,
                &emb(&[3]),
                1.0,
            )
            .unwrap();
        m.invalidate(id, 2.0).unwrap();
        assert!(m.current_facts(10).unwrap().is_empty());
        assert!(m.invalidate(id, 3.0).is_err());
    }

    #[test]
    fn refuses_a_different_embedder() {
        let dir = std::env::temp_dir().join(format!("grace-mem-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        drop(Memory::open(&dir, "a", 8).unwrap());
        assert!(Memory::open(&dir, "b", 8).is_err());
        assert!(Memory::open(&dir, "a", 8).is_ok());
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn wrong_dimension_is_an_error() {
        let mut m = mem();
        assert!(
            m.add_fact("x", FactKind::User, 0.5, None, &[0.0; 3], 0.0)
                .is_err()
        );
    }
}
