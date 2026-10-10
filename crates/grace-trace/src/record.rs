use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Span,
    Event,
}

/// One line of a trace file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub kind: Kind,
    /// Microseconds since the Unix epoch. For spans: when the span was opened.
    pub ts_us: u64,
    /// Spans only: how long the span was open.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dur_us: Option<u64>,
    pub service: String,
    pub name: String,
    pub target: String,
    pub level: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<u64>,
    /// Inherited from the nearest span (self included) that has a `turn` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub fields: Map<String, Value>,
}

/// Reads a trace file, skipping lines that don't parse (a crash can leave a torn last line).
pub fn read_session(path: &Path) -> anyhow::Result<Vec<Record>> {
    let reader = BufReader::new(File::open(path)?);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(record) = serde_json::from_str::<Record>(&line) {
            records.push(record);
        }
    }
    Ok(records)
}
