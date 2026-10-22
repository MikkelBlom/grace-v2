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

/// The hint must not hold up the reply: it runs next to memory recall (p50 124 ms, p90 285 ms
/// live) and is dropped if late. Live run 8 waited 392 ms p50 for a 6-way route decision.
pub const HINT_TIMEOUT: Duration = Duration::from_millis(300);
/// The lookup loop's own budget is 12 s; this only guards against a hung server.
pub const TOOL_TIMEOUT: Duration = Duration::from_secs(25);
/// Slow tools may run longer: run_code's sandbox allows up to 300 s, research about 3 minutes.
pub const SLOW_TOOL_TIMEOUT: Duration = Duration::from_secs(320);
/// Checker probability of "search" above which the user message gets the hint (fixtures/decide:
/// Gemma searched 17/17 lookup requests with a hint, 14/17 without, 0/33 others either way).
pub const HINT_AT: f64 = 0.5;
// Tools are offered on every turn. Leaving them out of chat turns (a "none" route) changed the
// prompt right after the system message from turn to turn, and llama-server re-read the whole
// conversation each time (live run 8: up to 3,100 tokens, 1.5 s); chat bench 02 scored the same
// with and without it (123/132 hard).
pub const HINT: &str = "[This likely needs a web lookup.]";
/// Tool calls per reply. Mikkel wants her to work a task out herself (2026-10-22): find the
/// tool, find the file, read it, write the result, open it; find_tools + use_tool alone are two.
pub const MAX_CALLS: usize = 8;

/// One yes/no decision: shorter than a 6-way choice, so faster on the Arc (fixtures/decide:
/// 17/17 lookup requests searched with its hint, 0/33 others).
const LOOKUP_Q: &str = "Mikkel spoke this request to his voice assistant. Does answering it need a \
lookup (web, a store's hours, prices, news, anything current or obscure) rather than general \
knowledge, maths, advice or chat?";

/// What the checker thinks a request needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    /// Probability that it needs a web lookup (drives the hint).
    pub search: f64,
}

impl Route {
    pub fn hint(&self) -> bool {
        self.search > HINT_AT
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
         reaction. It stays in the conversation: answer follow-ups (where it came from, which \
         ones, open that) from it, and only open links it contains.\n{result}"
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
    /// A destructive call waiting for his yes (ADR 0021): only his next utterance can answer it.
    pending: Mutex<Option<Pending>>,
}

#[derive(Debug, Clone)]
struct Pending {
    token: String,
    name: String,
    args: Value,
    summary: String,
}

/// His answer to "I'll delete the 3 files. Okay?": Some(true) only for a short, plain yes with
/// no "no" in it; anything else is not a yes, so the model can never talk itself into one.
pub fn answer_to(said: &str) -> Option<bool> {
    let lower = said.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != 'æ' && c != 'ø' && c != 'å')
        .filter(|w| !w.is_empty())
        .collect();
    let has = |set: &[&str]| words.iter().any(|w| set.contains(w));
    let phrase = |p: &str| format!(" {} ", words.join(" ")).contains(&format!(" {p} "));
    const NO: [&str; 9] = ["nej", "no", "nope", "ikke", "stop", "vent", "fortryd", "annuller", "wait"];
    if has(&NO) || phrase("lad være") || phrase("lad vær") {
        return Some(false);
    }
    const YES: [&str; 12] = [
        "ja", "jo", "jep", "jeps", "yes", "yeah", "yep", "okay", "ok", "kør", "sure", "selvfølgelig",
    ];
    let yes = has(&YES) || phrase("gør det") || phrase("go ahead") || phrase("det er fint")
        || phrase("fint nok") || phrase("bare gør");
    if yes && words.len() <= 8 { Some(true) } else { None }
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
            pending: Mutex::new(None),
        })
    }

    pub fn is_slow(&self, name: &str) -> bool {
        self.slow.contains(name)
    }

    /// Drops a question waiting for his yes (a new conversation in the sims).
    pub fn forget_pending(&self) {
        self.pending.lock().expect("pending poisoned").take();
    }

    /// Settles a destructive call waiting for his yes, from what he just said: runs it on a yes
    /// and returns the tool exchange for the model, or a note that nothing was done.
    async fn settle(&self, said: &str, span: &Span) -> Option<(Value, Value, Value)> {
        let p = self.pending.lock().expect("pending poisoned").take()?;
        let answer = answer_to(said);
        info!(parent: span, tool = %p.name, summary = %p.summary, ?answer, "confirmation");
        let out = if answer == Some(true) {
            call(&self.addr, json!({ "kind": "confirm", "token": p.token }))
                .await
                .unwrap_or_else(|e| json!({ "ok": false, "result": format!("It failed: {e}.") }))
        } else {
            json!({ "ok": false, "result": format!(
                "He did not say yes to: {}. Nothing was done; don't do it unless he asks again.", p.summary) })
        };
        Some((json!(p.name), p.args, out))
    }

    /// What the checker thinks `said` needs. Requests share one key, so the server drops an
    /// older transcript's check once a newer one waits (live run 7: 1 in 4 hints came late).
    pub async fn route(&self, said: &str) -> anyhow::Result<Route> {
        let v = call(
            &self.addr,
            json!({
                "kind": "decide",
                "key": "route",
                "state": format!("Mikkel said: \"{said}\""),
                "question": { "type": "noul", "instructions": LOOKUP_Q },
            }),
        )
        .await?;
        Ok(Route {
            search: v["answer"]["noul"].as_f64().context("no probability")?,
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
            Err(e) if e.to_string().contains("superseded") => None,
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
                info!(parent: span, search = r.search, "lookup check");
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

/// Drops the oldest half of the conversation between the system message and the current user
/// message, starting again on a user turn. False if there is nothing left to drop.
fn shrink(messages: &mut Vec<Value>) -> bool {
    let Some(last_user) = messages.iter().rposition(|m| m["role"] == "user") else {
        return false;
    };
    if last_user <= 1 {
        return false;
    }
    let mut cut = 1 + (last_user - 1).div_ceil(2);
    while cut < last_user && messages[cut]["role"] != "user" {
        cut += 1;
    }
    messages.drain(1..cut);
    true
}

/// A sentence that says Grace did something on the PC ("I have opened Notepad", "Spotify is now
/// open"). Only acting verbs: "I found it on Fandom" after an earlier search is a fair answer.
pub fn claims_action(text: &str) -> bool {
    static CLAIM: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"(?i)\bI(?:'ve| have)?\s+(?:just\s+|now\s+)?(?:opened|launched|started|played|paused|skipped|turned (?:it |the volume )?(?:up|down)|muted)\b|\b(?:is|are) now (?:open|opened|playing|running)\b|\b(?:is|are) (?:playing|running)\b|\bhas been (?:opened|started|launched)\b",
        )
        .expect("claim regex")
    });
    CLAIM.is_match(text)
}

/// A sentence that promises a tool action instead of doing it ("Let me double check that for
/// you", "I'll look it up"): live run 9 promised to double-check three times and never called a
/// tool.
pub fn promises_action(text: &str) -> bool {
    static PROMISE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"(?i)\b(?:I(?:'ll| will| am going to|'m going to)|let me)\s+(?:now\s+|just\s+|quickly\s+)?(?:double[- ]?check|check|look(?: it)? up|look|search|find|open|list|play|count)\b",
        )
        .expect("promise regex")
    });
    PROMISE.is_match(text)
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
    // His answer to a destructive call she asked about last turn.
    if let Some(t) = tools
        && let Some((name, args, out)) = t.settle(said, span).await
    {
        let name = name.as_str().unwrap_or("tool").to_owned();
        sink.tool_done(&name, &args, &out);
        let id = "confirmed_0".to_owned();
        let last_user = req.messages.iter().rposition(|m| m["role"] == "user");
        let exchange = [
            json!({ "role": "assistant", "content": "", "tool_calls": [{ "id": id, "type": "function",
                    "function": { "name": name, "arguments": args.to_string() } }] }),
            json!({ "role": "tool", "tool_call_id": id,
                    "content": tool_message(&name, out["result"].as_str().unwrap_or("")) }),
        ];
        // After his answer, so it reads: he said yes, then it ran.
        if let Some(i) = last_user {
            req.messages.splice(i + 1..i + 1, exchange);
        }
    }
    let mut retried_claim = false;
    loop {
        // (id, name, arguments) per tool call index, assembled from their pieces.
        let mut calls: Vec<(String, String, String)> = Vec::new();
        // Until a tool has run, the first sentence is held back and checked: live runs 7 and 8
        // said "I have opened Notepad" / "the projects folder" without calling anything.
        let mut guard = tools.is_some() && calls_made == 0 && !retried_claim;
        let mut pending = String::new();
        // All of this pass's text so far: the check reads the whole first sentence, while its
        // clauses already go to the voice (tools sim: "Okay, I will double check that" passed
        // when only "Okay," was checked).
        let mut seen = String::new();
        let mut claimed = false;
        let mut suspect = false;
        let mut opened = llm.stream(&req).await;
        // Live run 7: after 17 minutes the request outgrew the context and every reply after
        // that was silence. Drop the oldest half of the conversation and try again.
        for _ in 0..3 {
            match &opened {
                Err(e) if e.to_string().contains("exceed") => {
                    if !shrink(&mut req.messages) {
                        break;
                    }
                    warn!(parent: span, messages = req.messages.len(), "context exceeded, dropped the oldest turns");
                    opened = llm.stream(&req).await;
                }
                _ => break,
            }
        }
        match opened {
            Ok(mut stream) => loop {
                let delta = tokio::select! {
                    d = stream.next() => d,
                    _ = sink.cancelled() => None,
                };
                if sink.is_cancelled() {
                    return;
                }
                match delta {
                    Some(Ok(Delta::Content(text))) if guard => {
                        pending.push_str(&text);
                        seen.push_str(&text);
                        if suspect {
                            continue;
                        }
                        if claims_action(&seen) || promises_action(&seen) {
                            // Hold the rest to the end: a tool call may still follow the words.
                            suspect = true;
                            continue;
                        }
                        // Clauses go to the voice as they finish (waiting for the full stop
                        // delayed the first audio: live run 9, TTFT p50 279 -> 518 ms); the
                        // check stops after the first sentence.
                        if pending.contains(['.', '!', '?', ',', ';', ':']) || pending.len() > 48 {
                            sink.text(&std::mem::take(&mut pending));
                        }
                        // Two sentences: "Okay. Let me double check that for you." (tools sim).
                        if seen.matches(['.', '!', '?']).count() >= 2 || seen.len() > 200 {
                            guard = false;
                            sink.text(&std::mem::take(&mut pending));
                        }
                    }
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
        if guard && !seen.is_empty() {
            let unbacked = claims_action(&seen) || promises_action(&seen);
            if unbacked && calls.iter().all(|c| c.1.is_empty()) {
                claimed = true;
            } else if !(unbacked && calls.iter().any(|c| !c.1.is_empty())) {
                // A promise followed by its tool call: the filler line says it instead.
                sink.text(&std::mem::take(&mut pending));
            }
        }
        if claimed {
            warn!(parent: span, text = %seen, "claimed an action without a tool, asking again");
            retried_claim = true;
            // Words already spoken ("Okay.") and the new answer need a space between them.
            if seen.len() > pending.len() {
                sink.text(" ");
            }
            if let Some(last) = req.messages.iter_mut().rev().find(|m| m["role"] == "user") {
                let content = last["content"].as_str().unwrap_or("").to_owned();
                last["content"] = json!(format!(
                    "{content}\n\n[Nothing has been done yet, and promising doesn't do it. Call the tool now; if no tool can do it, say so.]"
                ));
            }
            continue;
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
            let mut parsed: Value = serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
            // The calculator reads his own transcribed words: Gemma rewrote "12.213" as
            // "tolv tusinde to hundrede treogtyve" (grace-tools-sim, live run 7).
            if name == "calculate"
                && let Some(o) = parsed.as_object_mut()
            {
                o.insert("heard".into(), json!(said));
            }
            let inner = if name == "use_tool" { parsed["name"].as_str().unwrap_or("") } else { &name };
            if tools.is_slow(inner) && sink.silent() {
                sink.filler(filler(reply_id, &name));
            }
            let stage = info_span!(parent: span, "stage", stage = "tool", model = name.as_str());
            let out = tokio::select! {
                r = tokio::time::timeout(
                    if tools.is_slow(inner) { SLOW_TOOL_TIMEOUT } else { TOOL_TIMEOUT },
                    tools.run(&name, &parsed),
                ) => r,
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
            if out["needs_confirm"] == true {
                // She asks now; it runs only on his next utterance's yes.
                *tools.pending.lock().expect("pending poisoned") = Some(Pending {
                    token: out["confirm"]["token"].as_str().unwrap_or("").to_owned(),
                    name: if name == "use_tool" { parsed["name"].as_str().unwrap_or("").to_owned() } else { name.clone() },
                    args: if name == "use_tool" { parsed["args"].clone() } else { parsed.clone() },
                    summary: out["confirm"]["summary"].as_str().unwrap_or("").to_owned(),
                });
                calls_made = MAX_CALLS;
            }
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

/// A tool call kept in the conversation history (role "tool"): live run 7 asked "which folders
/// start with P?", "where did you read that?" and "open that article" right after tool calls,
/// and without the results Grace made the answers (and an article URL) up.
pub fn tool_note(name: &str, args: &Value, out: &Value) -> String {
    let result: String = out["result"]
        .as_str()
        .unwrap_or("")
        .chars()
        .take(1500)
        .collect();
    json!({ "name": name, "args": args, "result": result }).to_string()
}

/// Chat messages for a history of ("user" | "assistant" | "tool", text): a tool note becomes the
/// assistant's tool call and its result, as Gemma saw them when it happened.
pub fn to_messages(history: &[(&str, String)]) -> Vec<Value> {
    let mut out = Vec::new();
    for (i, (role, text)) in history.iter().enumerate() {
        if *role != "tool" {
            out.push(json!({ "role": role, "content": text }));
            continue;
        }
        let v: Value = serde_json::from_str(text).unwrap_or(Value::Null);
        let name = v["name"].as_str().unwrap_or("tool");
        let id = format!("past_{i}");
        out.push(json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{ "id": id, "type": "function",
                             "function": { "name": name, "arguments": v["args"].to_string() } }],
        }));
        out.push(json!({
            "role": "tool",
            "tool_call_id": id,
            "content": tool_message(name, v["result"].as_str().unwrap_or("")),
        }));
    }
    out
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
    fn tool_results_stay_in_the_conversation() {
        let note = tool_note(
            "list_folder",
            &json!({ "folder": "Desktop" }),
            &json!({ "ok": true, "result": "Desktop: 17 folders. Folders: Portfolio, Projects" }),
        );
        let history = vec![
            ("user", "hvor mange mapper".to_owned()),
            ("tool", note),
            ("assistant", "17.".to_owned()),
        ];
        let m = to_messages(&history);
        assert_eq!(m.len(), 4);
        assert_eq!(m[1]["tool_calls"][0]["function"]["name"], "list_folder");
        assert_eq!(m[2]["role"], "tool");
        assert_eq!(m[2]["tool_call_id"], m[1]["tool_calls"][0]["id"]);
        assert!(m[2]["content"].as_str().unwrap().contains("Portfolio"));
    }

    #[test]
    fn shrink_keeps_system_and_current_turn_and_starts_on_user() {
        let mut m: Vec<Value> = std::iter::once(json!({ "role": "system" }))
            .chain((0..10).flat_map(|i| {
                [
                    json!({ "role": "user", "content": i }),
                    json!({ "role": "assistant", "content": i }),
                ]
            }))
            .chain(std::iter::once(json!({ "role": "user", "content": "now" })))
            .collect();
        assert!(shrink(&mut m));
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[1]["role"], "user");
        assert_eq!(m.last().unwrap()["content"], "now");
        assert!(m.len() < 22);
        let mut only = vec![json!({ "role": "system" }), json!({ "role": "user" })];
        assert!(!shrink(&mut only));
    }

    #[test]
    fn claims_are_acting_verbs_only() {
        assert!(claims_action("I have opened Notepad for you."));
        assert!(claims_action("Notepad is now open."));
        assert!(claims_action("I opened the projects folder."));
        assert!(claims_action("Spotify is playing."));
        assert!(!claims_action("I found that on The Mentalist Wiki."));
        assert!(!claims_action("I can open that for you."));
        assert!(!claims_action("The store is open until ten."));
        assert!(promises_action("Let me double check that for you."));
        assert!(promises_action("I will look it up."));
        assert!(promises_action("I'll open it now."));
        assert!(!promises_action("I will keep it shorter."));
        assert!(!promises_action("Let me know what you think."));
    }

    #[test]
    fn only_a_plain_yes_confirms() {
        for yes in ["ja", "Ja tak.", "jep gør det", "okay", "ja det er fint", "yes go ahead", "bare gør det"] {
            assert_eq!(answer_to(yes), Some(true), "{yes}");
        }
        for no in ["nej", "nej lad være", "ja nej vent", "stop", "ikke endnu", "no"] {
            assert_eq!(answer_to(no), Some(false), "{no}");
        }
        for neither in ["hvad er klokken", "hvilke filer er det", "",
                        "ja men først skal du lige fortælle mig hvad der ligger i mappen"] {
            assert_eq!(answer_to(neither), None, "{neither}");
        }
    }

    #[test]
    fn fillers_fit_the_tool() {
        assert!(
            filler(0, "find_files").contains("look") || filler(0, "find_files").contains("find")
        );
        assert_eq!(filler(1, "web_search"), "One sec, I'll look it up.");
    }
}
