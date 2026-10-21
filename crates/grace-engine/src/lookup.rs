//! Web lookup for Grace's replies (ADR 0019): the `web_search` tool Gemma can call, the
//! "needs a lookup?" hint from the decision checker, and the client for
//! `services/lookup_server.py` (JevK5 4B on the Arc iGPU + the search loop).
//!
//! Measured on fixtures/decide (docs/MODELS.md): Gemma alone searched on 14/17 lookup requests
//! and 0/33 others; with the checker's hint on the user message, 17/17 and 0/33.

use std::time::Duration;

use anyhow::{Context, bail};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
};

pub const TOOL_NAME: &str = "web_search";
/// The hint must not hold up the reply: it runs next to memory recall and is dropped if late.
pub const HINT_TIMEOUT: Duration = Duration::from_millis(400);
/// The loop's own budget is 12 s; this only guards against a hung server.
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(20);
/// Checker probability above which the user message gets the hint.
pub const HINT_AT: f64 = 0.5;
pub const HINT: &str = "[This likely needs a web lookup.]";

const LOOKUP_Q: &str = "Mikkel spoke this request to his voice assistant. Does answering it need a \
lookup (web, a store's hours, prices, news, anything current or obscure) rather than general \
knowledge, maths, advice or chat?";

/// The tool definition sent with every reply request.
pub fn tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Search the web and read the best pages. Returns what the sources say.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "Mikkel's question, in English, as one full sentence." },
                    "query": { "type": "string", "description": "Short, specific English search query." }
                },
                "required": ["query"]
            }
        }
    })
}

/// What Grace says while she searches, varied by reply.
pub fn filler(reply: u64) -> &'static str {
    const LINES: [&str; 4] = [
        "Let me check.",
        "One sec, I'll look it up.",
        "Let me look that up.",
        "Checking.",
    ];
    LINES[(reply as usize) % LINES.len()]
}

/// The tool result Gemma answers from: the evidence, without URLs (they'd be read aloud).
pub fn result_text(out: &Value) -> String {
    let evidence = out["evidence"].as_array().cloned().unwrap_or_default();
    let mut s = match out["status"].as_str() {
        Some("answered") => String::from("Found it. Sources:\n"),
        _ if evidence.is_empty() => {
            return "Nothing reliable was found. Say you couldn't find it.".into();
        }
        _ => String::from(
            "No source clearly answered it. Partial matches (say you couldn't confirm it):\n",
        ),
    };
    for (i, e) in evidence.iter().take(3).enumerate() {
        let site = e["url"]
            .as_str()
            .and_then(|u| u.split('/').nth(2))
            .unwrap_or("");
        s.push_str(&format!(
            "{}. {} ({site}): {}\n",
            i + 1,
            e["title"].as_str().unwrap_or(""),
            e["text"].as_str().unwrap_or("").trim()
        ));
    }
    s
}

/// (question, query) from the tool call's JSON arguments.
pub fn arguments(args: &str, fallback_question: &str) -> (String, String) {
    let v: Value = serde_json::from_str(args).unwrap_or(Value::Null);
    let query = v["query"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(fallback_question)
        .to_owned();
    let question = v["question"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .map_or_else(|| query.clone(), str::to_owned);
    (question, query)
}

/// Client for `services/lookup_server.py`: one connection per request, so a hint never waits
/// behind a running search and a dropped (barged-in) search just closes its socket.
pub struct Lookup {
    addr: String,
}

impl Lookup {
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        TcpStream::connect(addr).await.with_context(|| {
            format!("lookup server not reachable at {addr}; start services/lookup_server.py")
        })?;
        Ok(Self { addr: addr.into() })
    }

    async fn call(&self, req: Value) -> anyhow::Result<Value> {
        let stream = TcpStream::connect(&self.addr).await?;
        stream.set_nodelay(true)?;
        let (r, mut w) = stream.into_split();
        w.write_all((req.to_string() + "\n").as_bytes()).await?;
        let mut line = String::new();
        BufReader::new(r).read_line(&mut line).await?;
        let v: Value = serde_json::from_str(&line).context("bad lookup reply")?;
        if let Some(e) = v["error"].as_str() {
            bail!("lookup server: {e}");
        }
        Ok(v)
    }

    /// The checker's probability that `said` needs a lookup.
    pub async fn needs_lookup(&self, said: &str) -> anyhow::Result<f64> {
        let v = self
            .call(json!({
                "kind": "decide",
                "state": format!("Mikkel said: \"{said}\""),
                "question": { "type": "noul", "instructions": LOOKUP_Q },
            }))
            .await?;
        v["answer"]["noul"].as_f64().context("no probability")
    }

    pub async fn search(&self, question: &str, query: &str) -> anyhow::Result<Value> {
        self.call(json!({ "kind": "lookup", "question": question, "query": query }))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_fall_back_sensibly() {
        assert_eq!(
            arguments(r#"{"query":"Sean Parker actor"}"#, "hvem spillede"),
            ("Sean Parker actor".into(), "Sean Parker actor".into())
        );
        assert_eq!(
            arguments(r#"{"question":"Who played him?","query":"x"}"#, ""),
            ("Who played him?".into(), "x".into())
        );
        assert_eq!(
            arguments("not json", "hvem spillede"),
            ("hvem spillede".into(), "hvem spillede".into())
        );
    }

    #[test]
    fn result_has_no_urls_and_says_when_unsure() {
        let found = json!({ "status": "answered", "evidence": [
            { "title": "Sean Parker", "url": "https://en.wikipedia.org/wiki/Sean_Parker", "text": "Played by Justin Timberlake.", "p": 0.95 }
        ]});
        let s = result_text(&found);
        assert!(s.starts_with("Found it."));
        assert!(s.contains("(en.wikipedia.org)") && !s.contains("https://"));
        let partial = json!({ "status": "not_found", "evidence": [
            { "title": "t", "url": "https://x.dk/a", "text": "maybe", "p": 0.6 }
        ]});
        assert!(result_text(&partial).contains("couldn't confirm"));
        assert!(
            result_text(&json!({ "status": "not_found", "evidence": [] }))
                .contains("couldn't find")
        );
    }
}
