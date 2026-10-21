//! Grace's tools (ADR 0019, ADR 0020): the client for `services/tools_server.py`, the
//! checker's "needs a lookup?" hint, and the reply loop that lets Gemma call tools. The live
//! engine (`conductor::speak_reply`) and `grace-tools-sim` both run [`reply`], so the sim
//! measures exactly what Grace does.
//!
//! Tools come from the server's manifest: web_search, open_app, open_url, find_files,
//! list_folder, open_file. None deletes, sends or buys anything. Tools only run once the reply
//! is committed (a prepared reply that gets thrown away never opens a program or searches).

use std::{
    collections::{HashMap, HashSet},
    future::Future,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, bail};
use futures_util::StreamExt;
use grace_llm::client::{ChatRequest, Delta, LlmClient};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
};
use tracing::{Span, info, info_span, warn};

/// The hint must not hold up the reply: it runs next to memory recall and is dropped if late.
pub const HINT_TIMEOUT: Duration = Duration::from_millis(400);
/// The lookup loop's own budget is 12 s; this only guards against a hung server.
pub const TOOL_TIMEOUT: Duration = Duration::from_secs(25);
/// Checker probability of "search" above which the user message gets the hint (fixtures/decide:
/// Gemma searched 17/17 lookup requests with a hint, 14/17 without, 0/33 others either way).
pub const HINT_AT: f64 = 0.5;
/// Checker probability of "no tool" above which the request goes without tool definitions:
/// with them in the prompt, Gemma asked for facts her memory already listed (chat bench 02:
/// 131/132 -> 123/132 hard checks). At 0.8 no tool request of 45 was routed to "none", and 24 of
/// 39 chat turns went without tools (fixtures/decide + fixtures/tools + chat lines, 2026-10-21).
pub const SKIP_TOOLS_AT: f64 = 0.8;
pub const HINT: &str = "[This likely needs a web lookup.]";
/// Tool calls per reply: find a file, then open it; search, then search again.
pub const MAX_CALLS: usize = 3;

const ROUTE_Q: &str = "Mikkel said this to his voice assistant. Which tool does it need, if any?";

/// The checker's options for [`ROUTE_Q`].
fn route_options() -> Value {
    json!({
        "search": "Look something up: news, facts it may not know, a film or show, opening hours, prices, current events.",
        "weather": "The weather forecast.",
        "calculate": "Arithmetic or maths with numbers.",
        "open": "Open a program, app or web page.",
        "files": "Find, list or open files and folders on his computer.",
        "none": "No tool: chat, opinions, advice, general knowledge, things about Mikkel, or something no tool can do (messages, timers).",
    })
}

/// What the checker thinks a request needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    /// Probability that it needs a web lookup (drives the hint).
    pub search: f64,
    /// Probability that it needs no tool at all (drives leaving the tools out).
    pub none: f64,
}

impl Route {
    pub fn hint(&self) -> bool {
        self.search > HINT_AT
    }
    pub fn offer_tools(&self) -> bool {
        self.none < SKIP_TOOLS_AT
    }
}

/// What Grace says while a slow tool runs, varied by reply.
pub fn filler(reply: u64, tool: &str) -> &'static str {
    const SEARCH: [&str; 4] = [
        "Let me check.",
        "One sec, I'll look it up.",
        "Let me look that up.",
        "Checking.",
    ];
    const FILES: [&str; 2] = ["Let me look.", "One sec, I'll find it."];
    if tool == "find_files" {
        FILES[(reply as usize) % FILES.len()]
    } else {
        SEARCH[(reply as usize) % SEARCH.len()]
    }
}

/// The tool message Gemma answers from. Live run 6: a bare search result read like something
/// Mikkel said ("That sounds intense. So Jane got kidnapped?"), so it says whose it is.
pub fn tool_message(name: &str, result: &str) -> String {
    format!(
        "Result of your {name} call. This is what you found or did, not something Mikkel said. \
         Tell him the result directly in one or two plain sentences, as information, not as a \
         reaction.\n{result}"
    )
}

/// The client for `services/tools_server.py`: one connection per request, so a hint never waits
/// behind a running search, and a dropped (barged-in) call closes its socket, which the server
/// notices and stops.
pub struct Tools {
    addr: String,
    pub manifest: Vec<Value>,
    slow: HashSet<String>,
    /// The needs-lookup answer per transcript, started as soon as the words are transcribed:
    /// right after Gemma has generated, the shared power budget slows the Arc to 290-390 ms per
    /// decision, too close to the reply's 400 ms cap (grace-tools-sim v2: 14/48 hints late).
    hints: Mutex<HashMap<String, Arc<tokio::sync::OnceCell<Option<Route>>>>>,
}

impl Tools {
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        let v = call(addr, json!({ "kind": "manifest" }))
            .await
            .with_context(|| {
                format!("tools server not reachable at {addr}; start services/tools_server.py")
            })?;
        let manifest = v["tools"].as_array().cloned().unwrap_or_default();
        let slow = v["slow"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s.as_str().map(str::to_owned))
            .collect();
        Ok(Self {
            addr: addr.into(),
            manifest,
            slow,
            hints: Mutex::new(HashMap::new()),
        })
    }

    pub fn is_slow(&self, name: &str) -> bool {
        self.slow.contains(name)
    }

    /// What the checker thinks `said` needs.
    pub async fn route(&self, said: &str) -> anyhow::Result<Route> {
        let v = call(
            &self.addr,
            json!({
                "kind": "decide",
                "state": format!("Mikkel said: \"{said}\""),
                "question": { "type": "choice", "instructions": ROUTE_Q, "criteria": route_options() },
            }),
        )
        .await?;
        let p = &v["answer"]["probabilities"];
        Ok(Route {
            search: p["search"].as_f64().context("no probability")?,
            none: p["none"].as_f64().context("no probability")?,
        })
    }

    pub async fn run(&self, name: &str, args: &Value) -> anyhow::Result<Value> {
        call(
            &self.addr,
            json!({ "kind": "tool", "name": name, "args": args }),
        )
        .await
    }

    fn hint_cell(&self, said: &str) -> Arc<tokio::sync::OnceCell<Option<Route>>> {
        let mut hints = self.hints.lock().expect("hints poisoned");
        if hints.len() > 64 {
            hints.clear();
        }
        hints.entry(said.to_owned()).or_default().clone()
    }

    async fn checked(&self, said: &str) -> Option<Route> {
        match self.route(said).await {
            Ok(r) => Some(r),
            Err(e) => {
                warn!(error = %e, "lookup check failed");
                None
            }
        }
    }

    /// Starts the needs-lookup check for a fresh transcript, so the reply finds it done.
    pub fn prefetch(self: &Arc<Self>, said: &str) {
        let cell = self.hint_cell(said);
        let (me, said) = (self.clone(), said.to_owned());
        tokio::spawn(async move {
            cell.get_or_init(|| me.checked(&said)).await;
        });
    }

    /// The checker's route for the user message, or None (no server or too slow). Waits at
    /// most [`HINT_TIMEOUT`] for a check that `prefetch` may have started earlier.
    pub async fn hint(&self, said: &str, span: &Span) -> Option<Route> {
        let _stage =
            info_span!(parent: span, "stage", stage = "lookup.check", model = "jevk5-4b-arc");
        let cell = self.hint_cell(said);
        match tokio::time::timeout(HINT_TIMEOUT, cell.get_or_init(|| self.checked(said))).await {
            Ok(Some(r)) => {
                info!(parent: span, search = r.search, none = r.none, "lookup check");
                Some(*r)
            }
            Ok(None) => None,
            Err(_) => {
                warn!(parent: span, "lookup check too slow, skipped");
                None
            }
        }
    }
}

async fn call(addr: &str, req: Value) -> anyhow::Result<Value> {
    let stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    let (r, mut w) = stream.into_split();
    w.write_all((req.to_string() + "\n").as_bytes()).await?;
    let mut line = String::new();
    BufReader::new(r).read_line(&mut line).await?;
    let v: Value = serde_json::from_str(&line).context("bad tools server reply")?;
    if let Some(e) = v["error"].as_str() {
        bail!("tools server: {e}");
    }
    Ok(v)
}

/// What the reply loop needs from its caller: where the words go and whether it may go on.
pub trait Sink {
    /// Streamed reply text.
    fn text(&mut self, text: &str);
    /// A line to say while a slow tool runs.
    fn filler(&mut self, line: &str);
    /// A tool was called and returned (for traces and the sim).
    fn tool_done(&mut self, name: &str, args: &Value, out: &Value);
    /// Resolves once the reply is cancelled (barge-in); never, for the sim.
    fn cancelled(&mut self) -> impl Future<Output = ()> + Send;
    fn is_cancelled(&self) -> bool;
    /// Waits until the reply is committed; false if it was thrown away instead.
    fn ready_to_act(&mut self) -> impl Future<Output = bool> + Send;
    /// True while nothing has been said yet in this reply.
    fn silent(&self) -> bool;
}

/// One reply with tools: stream Gemma's answer; when she calls tools, run them (after the commit),
/// hand back the results and stream again. At most [`MAX_CALLS`] calls; the last pass has no tools.
pub async fn reply<S: Sink + Send>(
    llm: &LlmClient,
    tools: Option<&Tools>,
    mut req: ChatRequest,
    reply_id: u64,
    said: &str,
    span: &Span,
    sink: &mut S,
) {
    if let Some(t) = tools {
        req.tools = t.manifest.clone();
    }
    let mut calls_made = 0;
    loop {
        // (id, name, arguments) per tool call index, assembled from their pieces.
        let mut calls: Vec<(String, String, String)> = Vec::new();
        match llm.stream(&req).await {
            Ok(mut stream) => loop {
                let delta = tokio::select! {
                    d = stream.next() => d,
                    _ = sink.cancelled() => None,
                };
                if sink.is_cancelled() {
                    return;
                }
                match delta {
                    Some(Ok(Delta::Content(text))) => sink.text(&text),
                    Some(Ok(Delta::ToolCall {
                        index,
                        id,
                        name,
                        arguments,
                    })) => {
                        if calls.len() <= index {
                            calls.resize(index + 1, Default::default());
                        }
                        let c = &mut calls[index];
                        if let Some(id) = id {
                            c.0 = id;
                        }
                        if let Some(name) = name {
                            c.1 = name;
                        }
                        c.2.push_str(&arguments);
                    }
                    Some(Ok(Delta::Done { .. })) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        warn!(parent: span, error = %e, "llm stream error");
                        break;
                    }
                }
            },
            Err(e) => {
                warn!(parent: span, error = %e, "llm request failed");
                return;
            }
        }
        calls.retain(|c| !c.1.is_empty());
        let Some(tools) = tools else { return };
        if calls.is_empty() || calls_made >= MAX_CALLS {
            return;
        }
        // Nothing acts or searches for a reply that may still be thrown away.
        if !sink.ready_to_act().await {
            return;
        }
        let mut tool_calls = Vec::new();
        let mut results = Vec::new();
        for (i, (id, name, args)) in calls.into_iter().enumerate() {
            if calls_made >= MAX_CALLS {
                break;
            }
            calls_made += 1;
            let id = if id.is_empty() {
                format!("call_{calls_made}_{i}")
            } else {
                id
            };
            let parsed: Value = serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
            if tools.is_slow(&name) && sink.silent() {
                sink.filler(filler(reply_id, &name));
            }
            let stage = info_span!(parent: span, "stage", stage = "tool", model = name.as_str());
            let out = tokio::select! {
                r = tokio::time::timeout(TOOL_TIMEOUT, tools.run(&name, &parsed)) => r,
                _ = sink.cancelled() => return,
            };
            drop(stage);
            let out = match out {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    warn!(parent: span, tool = %name, error = %e, "tool failed");
                    json!({ "ok": false, "result": format!("The {name} tool failed. Say it didn't work.") })
                }
                Err(_) => {
                    warn!(parent: span, tool = %name, "tool timed out");
                    json!({ "ok": false, "result": format!("The {name} tool took too long. Say it didn't work this time.") })
                }
            };
            info!(parent: span, tool = %name, args = %parsed, ok = %out["ok"], status = %out["status"], ms = %out["ms"], said, "tool call");
            sink.tool_done(&name, &parsed, &out);
            tool_calls.push(json!({ "id": id, "type": "function",
                                    "function": { "name": name, "arguments": args } }));
            results.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": tool_message(&name, out["result"].as_str().unwrap_or("")),
            }));
        }
        req.messages
            .push(json!({ "role": "assistant", "content": "", "tool_calls": tool_calls }));
        req.messages.extend(results);
        if calls_made >= MAX_CALLS {
            req.tools.clear();
        }
    }
}

/// A sink that only collects: for benches (no audio, never cancelled, always committed).
#[derive(Default)]
pub struct Collect {
    pub text: String,
    /// (tool, arguments, result) per call.
    pub calls: Vec<(String, Value, Value)>,
}

impl Sink for Collect {
    fn text(&mut self, text: &str) {
        self.text.push_str(text);
    }
    fn filler(&mut self, line: &str) {
        self.text.push_str(line);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_message_says_whose_words_these_are() {
        let m = tool_message("web_search", "Found. Justin Timberlake played Sean Parker.");
        assert!(m.contains("not something Mikkel said"));
        assert!(m.ends_with("Justin Timberlake played Sean Parker."));
    }

    #[test]
    fn fillers_fit_the_tool() {
        assert!(
            filler(0, "find_files").contains("look") || filler(0, "find_files").contains("find")
        );
        assert_eq!(filler(1, "web_search"), "One sec, I'll look it up.");
    }
}
