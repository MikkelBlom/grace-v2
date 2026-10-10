//! LLM latency bench against a running llama-server: time to first token, time to first
//! speakable piece (what the user waits for), decode speed, prefix-cache reuse, and whether the
//! model picked the expected tool or answered directly.

use std::{path::Path, time::Instant};

use anyhow::Context;
use futures_util::StreamExt;
use grace_llm::{
    client::{ChatRequest, Delta, LlmClient},
    sentences::SentenceSplitter,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::metrics::percentile;

#[derive(Deserialize)]
struct Prompt {
    id: String,
    text: String,
    /// `chat` (answer directly) or `tool:<name>`.
    expect: String,
}

#[derive(Serialize)]
struct Item {
    id: String,
    expect: String,
    got: String,
    correct: bool,
    ttft_ms: Option<f64>,
    first_piece_ms: Option<f64>,
    total_ms: f64,
    reply: String,
    first_piece: Option<String>,
    tool_args: Option<String>,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    prompt_ms: Option<f64>,
    decode_tok_s: Option<f64>,
}

pub struct Options<'a> {
    pub url: &'a str,
    pub dir: &'a Path,
    pub label: &'a str,
    pub out: &'a Path,
    pub repeat: usize,
}

pub fn run(opts: Options) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(run_async(opts))
}

async fn run_async(opts: Options<'_>) -> anyhow::Result<()> {
    let system = std::fs::read_to_string(opts.dir.join("system.md")).context("system.md")?;
    let tools: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(opts.dir.join("tools.json"))?)?;
    let prompts: Vec<Prompt> = std::fs::read_to_string(opts.dir.join("prompts.jsonl"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;

    let client = LlmClient::new(opts.url);
    anyhow::ensure!(
        client.healthy().await,
        "llama-server not healthy at {}",
        opts.url
    );

    // Warm-up: fills the KV cache with the shared prefix, as a running Grace would have.
    let warm = request(&system, &tools, "Hej.");
    drain(&client, &warm).await?;

    let mut items = Vec::new();
    for round in 0..opts.repeat {
        for p in &prompts {
            let item = one(&client, &system, &tools, p).await?;
            eprintln!(
                "[{}] r{round} {:<16} {:<22} ttft={:>5.0} first={:>5.0} total={:>5.0} {}",
                opts.label,
                p.id,
                item.got,
                item.ttft_ms.unwrap_or(f64::NAN),
                item.first_piece_ms.unwrap_or(f64::NAN),
                item.total_ms,
                if item.correct { "ok" } else { "MISS" }
            );
            items.push(item);
        }
    }

    let col = |f: fn(&Item) -> Option<f64>| -> Vec<f64> { items.iter().filter_map(f).collect() };
    let chat_first: Vec<f64> = items
        .iter()
        .filter(|i| i.expect == "chat")
        .filter_map(|i| i.first_piece_ms)
        .collect();
    let summary = json!({
        "kind": "llm",
        "label": opts.label,
        "requests": items.len(),
        "accuracy": items.iter().filter(|i| i.correct).count() as f64 / items.len() as f64,
        "ttft_p50_ms": percentile(&col(|i| i.ttft_ms), 50.0),
        "ttft_p95_ms": percentile(&col(|i| i.ttft_ms), 95.0),
        "chat_first_piece_p50_ms": percentile(&chat_first, 50.0),
        "chat_first_piece_p95_ms": percentile(&chat_first, 95.0),
        "decode_tok_s_p50": percentile(&col(|i| i.decode_tok_s), 50.0),
        "prompt_ms_p50": percentile(&col(|i| i.prompt_ms), 50.0),
        "url": opts.url,
    });
    crate::write_report(opts.out, "llm", opts.label, &summary, &items)
}

fn request(system: &str, tools: &[Value], user: &str) -> ChatRequest {
    let mut req = ChatRequest::new(vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user}),
    ]);
    req.tools = tools.to_vec();
    req.max_tokens = 256;
    req
}

async fn drain(client: &LlmClient, req: &ChatRequest) -> anyhow::Result<()> {
    let mut s = client.stream(req).await?;
    while let Some(d) = s.next().await {
        d?;
    }
    Ok(())
}

async fn one(
    client: &LlmClient,
    system: &str,
    tools: &[Value],
    p: &Prompt,
) -> anyhow::Result<Item> {
    let req = request(system, tools, &p.text);
    let started = Instant::now();
    let ms = || started.elapsed().as_secs_f64() * 1000.0;
    let mut stream = client.stream(&req).await?;
    let mut splitter = SentenceSplitter::default();
    let (mut ttft, mut first_ms, mut first_piece) = (None, None, None);
    let mut reply = String::new();
    let (mut tool, mut args) = (None::<String>, String::new());
    let mut timings = None;
    while let Some(delta) = stream.next().await {
        match delta? {
            Delta::Content(text) => {
                ttft.get_or_insert_with(ms);
                reply.push_str(&text);
                if let Some(piece) = splitter.push(&text).into_iter().next()
                    && first_piece.is_none()
                {
                    first_ms = Some(ms());
                    first_piece = Some(piece);
                }
            }
            Delta::ToolCall {
                name, arguments, ..
            } => {
                ttft.get_or_insert_with(ms);
                if let Some(n) = name {
                    tool = Some(n);
                }
                args.push_str(&arguments);
            }
            Delta::Reasoning(_) => {}
            Delta::Done { timings: t, .. } => timings = t,
        }
    }
    let total = ms();
    if first_piece.is_none()
        && let Some(rest) = splitter.finish()
    {
        first_ms = Some(total);
        first_piece = Some(rest);
    }
    let got = tool
        .as_ref()
        .map_or("chat".to_owned(), |t| format!("tool:{t}"));
    let num = |k: &str| {
        timings
            .as_ref()
            .and_then(|t| t.get(k))
            .and_then(Value::as_f64)
    };
    Ok(Item {
        id: p.id.clone(),
        expect: p.expect.clone(),
        correct: got == p.expect,
        got,
        ttft_ms: ttft,
        first_piece_ms: first_ms,
        total_ms: total,
        reply,
        first_piece,
        tool_args: tool.map(|_| args),
        prompt_tokens: num("prompt_n").map(|v| v as u64),
        cached_tokens: num("cache_n").map(|v| v as u64),
        prompt_ms: num("prompt_ms"),
        decode_tok_s: num("predicted_per_second"),
    })
}
