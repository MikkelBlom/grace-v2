//! Grace debug viewer: serves trace files as a live per-turn latency waterfall.
//!
//!   cargo run -p grace-debug -- --traces traces
//!
//! The UI is compiled in. Pass `--ui-dir crates/grace-debug/ui` to serve it from disk instead,
//! so viewer changes show up on browser refresh without rebuilding.

use std::{
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{Path as UrlPath, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::get,
};
use clap::Parser;
use grace_trace::Record;
use serde::{Deserialize, Serialize};

const EMBEDDED_UI: &str = include_str!("../ui/index.html");

#[derive(Parser)]
struct Args {
    /// Directory containing `*.jsonl` trace files.
    #[arg(long, default_value = "traces")]
    traces: PathBuf,
    #[arg(long, default_value_t = 7700)]
    port: u16,
    /// Serve `index.html` from this directory on every request (UI development).
    #[arg(long)]
    ui_dir: Option<PathBuf>,
}

struct AppState {
    traces: PathBuf,
    ui_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.traces)?;
    let state = Arc::new(AppState {
        traces: args.traces.clone(),
        ui_dir: args.ui_dir,
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/api/sessions", get(sessions))
        .route("/api/sessions/{name}", get(session))
        .with_state(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    println!(
        "grace-debug on http://{addr}  (traces: {})",
        args.traces.display()
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index(State(state): State<Arc<AppState>>) -> Response {
    match &state.ui_dir {
        Some(dir) => match tokio::fs::read_to_string(dir.join("index.html")).await {
            Ok(html) => Html(html).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        },
        None => Html(EMBEDDED_UI).into_response(),
    }
}

#[derive(Serialize)]
struct SessionInfo {
    name: String,
    bytes: u64,
    modified_ms: u64,
}

async fn sessions(State(state): State<Arc<AppState>>) -> Result<Json<Vec<SessionInfo>>, AppError> {
    let mut list = Vec::new();
    for entry in std::fs::read_dir(&state.traces)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(stem) = name.strip_suffix(".jsonl") else {
            continue;
        };
        let meta = entry.metadata()?;
        let modified_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_millis() as u64);
        list.push(SessionInfo {
            name: stem.to_owned(),
            bytes: meta.len(),
            modified_ms,
        });
    }
    list.sort_by_key(|s| std::cmp::Reverse(s.modified_ms));
    Ok(Json(list))
}

#[derive(Deserialize)]
struct Tail {
    /// Byte offset to resume from; the response returns the next one.
    #[serde(default)]
    offset: u64,
}

#[derive(Serialize)]
struct Chunk {
    records: Vec<Record>,
    offset: u64,
}

async fn session(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
    Query(tail): Query<Tail>,
) -> Result<Json<Chunk>, AppError> {
    let path = session_path(&state.traces, &name).ok_or(AppError::NotFound)?;
    let chunk = tokio::task::spawn_blocking(move || read_from(&path, tail.offset))
        .await
        .map_err(|e| AppError::Internal(e.into()))??;
    Ok(Json(chunk))
}

/// Only plain file stems inside the trace dir; no separators or traversal.
fn session_path(dir: &Path, name: &str) -> Option<PathBuf> {
    let safe = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && !name.contains("..");
    let path = dir.join(format!("{name}.jsonl"));
    (safe && path.is_file()).then_some(path)
}

/// Reads complete lines from `offset`. A trailing partial line (writer mid-flush) is left
/// for the next poll by not advancing the offset past it.
fn read_from(path: &Path, offset: u64) -> std::io::Result<Chunk> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let start = if offset > len { 0 } else { offset };
    file.seek(SeekFrom::Start(start))?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut pos = start;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 || !line.ends_with('\n') {
            break;
        }
        pos += n as u64;
        if let Ok(record) = serde_json::from_str::<Record>(line.trim_end()) {
            records.push(record);
        }
    }
    Ok(Chunk {
        records,
        offset: pos,
    })
}

enum AppError {
    NotFound,
    Internal(anyhow::Error),
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        Self::Internal(e.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            Self::NotFound => StatusCode::NOT_FOUND.into_response(),
            Self::Internal(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_traversal_and_unknown_names() {
        let dir = std::env::temp_dir();
        assert!(session_path(&dir, "../secret").is_none());
        assert!(session_path(&dir, "a/b").is_none());
        assert!(session_path(&dir, "").is_none());
        assert!(session_path(&dir, "does-not-exist-xyz").is_none());
    }

    #[test]
    fn tail_leaves_partial_line_for_next_poll() {
        let dir = tempfile_dir();
        let path = dir.join("s.jsonl");
        let mut f = File::create(&path).unwrap();
        let line =
            r#"{"kind":"event","ts_us":1,"service":"s","name":"a","target":"t","level":"INFO"}"#;
        write!(f, "{line}\n{{\"kind\":").unwrap();
        drop(f);
        let first = read_from(&path, 0).unwrap();
        assert_eq!(first.records.len(), 1);
        assert_eq!(first.offset, line.len() as u64 + 1);
        let again = read_from(&path, first.offset).unwrap();
        assert!(again.records.is_empty());
        assert_eq!(again.offset, first.offset);
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("grace-debug-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
