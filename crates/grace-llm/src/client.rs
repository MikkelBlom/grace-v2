//! Streaming client for llama-server's OpenAI-compatible `/v1/chat/completions`.
//!
//! Dropping the returned stream closes the HTTP connection, which makes llama-server stop
//! generating: that is how a reply is cancelled on barge-in.

use std::pin::Pin;

use anyhow::{Context, bail};
use futures_util::{Stream, StreamExt};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Content(String),
    /// Hidden reasoning text, if the model emits any (kept out of speech).
    Reasoning(String),
    /// A fragment of a tool call; `arguments` arrive in pieces and must be concatenated.
    ToolCall {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    /// Final chunk. `timings` is llama-server's own report (prompt/cache/decode numbers).
    Done {
        finish_reason: Option<String>,
        timings: Option<Value>,
    },
}

pub type DeltaStream = Pin<Box<dyn Stream<Item = anyhow::Result<Delta>> + Send>>;

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Extra top-level fields passed through verbatim (e.g. `cache_prompt`, `chat_template_kwargs`).
    pub extra: Map<String, Value>,
}

impl ChatRequest {
    pub fn new(messages: Vec<Value>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            max_tokens: 512,
            temperature: 0.6,
            extra: Map::new(),
        }
    }

    fn body(&self) -> Value {
        let mut body = json!({
            "messages": self.messages,
            "stream": true,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            // Reuse the KV cache for the shared prompt prefix (system prompt, tools, history).
            "cache_prompt": true,
        });
        if !self.tools.is_empty() {
            body["tools"] = Value::Array(self.tools.clone());
        }
        for (k, v) in &self.extra {
            body[k] = v.clone();
        }
        body
    }
}

#[derive(Debug, Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    base: String,
}

impl LlmClient {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base: base.into().trim_end_matches('/').to_owned(),
        }
    }

    pub async fn healthy(&self) -> bool {
        matches!(
            self.http.get(format!("{}/health", self.base)).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    /// One non-streamed completion whose content must match `schema` (grammar-constrained by
    /// llama-server), parsed as JSON. For background work such as memory extraction.
    pub async fn complete_json(&self, req: &ChatRequest, schema: &Value) -> anyhow::Result<Value> {
        let mut body = req.body();
        body["stream"] = json!(false);
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": { "name": "result", "schema": schema, "strict": true },
        });
        let resp = self
            .http
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&body)
            .send()
            .await
            .context("sending completion request")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("llama-server returned {status}: {text}");
        }
        let v: Value = resp.json().await?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .context("completion has no content")?;
        serde_json::from_str(content)
            .with_context(|| format!("completion is not valid JSON: {content}"))
    }

    pub async fn stream(&self, req: &ChatRequest) -> anyhow::Result<DeltaStream> {
        let resp = self
            .http
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&req.body())
            .send()
            .await
            .context("sending chat request")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("llama-server returned {status}: {text}");
        }
        let mut bytes = resp.bytes_stream();
        let stream = async_stream(move |tx| async move {
            let mut buf = Vec::new();
            while let Some(chunk) = bytes.next().await {
                buf.extend_from_slice(&chunk?);
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    for delta in parse_sse_line(&line)? {
                        if tx.send(Ok(delta)).await.is_err() {
                            return Ok(()); // receiver dropped: cancelled
                        }
                    }
                }
            }
            Ok(())
        });
        Ok(stream)
    }
}

/// Minimal channel-backed stream so the parser can be written as a plain async loop.
fn async_stream<F, Fut>(f: F) -> DeltaStream
where
    F: FnOnce(tokio::sync::mpsc::Sender<anyhow::Result<Delta>>) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let err_tx = tx.clone();
    let fut = f(tx);
    tokio::spawn(async move {
        if let Err(e) = fut.await {
            let _ = err_tx.send(Err(e)).await;
        }
    });
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    }))
}

/// Parses one SSE line (`data: {...}`) into zero or more deltas.
pub fn parse_sse_line(line: &[u8]) -> anyhow::Result<Vec<Delta>> {
    let line = std::str::from_utf8(line)?.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(Vec::new());
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(Vec::new());
    }
    let v: Value = serde_json::from_str(data).with_context(|| format!("bad SSE json: {data}"))?;
    let mut out = Vec::new();
    let choice = &v["choices"][0];
    let delta = &choice["delta"];
    if let Some(text) = delta["reasoning_content"]
        .as_str()
        .filter(|s| !s.is_empty())
    {
        out.push(Delta::Reasoning(text.to_owned()));
    }
    if let Some(text) = delta["content"].as_str().filter(|s| !s.is_empty()) {
        out.push(Delta::Content(text.to_owned()));
    }
    if let Some(calls) = delta["tool_calls"].as_array() {
        for call in calls {
            out.push(Delta::ToolCall {
                index: call["index"].as_u64().unwrap_or(0) as usize,
                id: call["id"].as_str().map(str::to_owned),
                name: call["function"]["name"].as_str().map(str::to_owned),
                arguments: call["function"]["arguments"]
                    .as_str()
                    .unwrap_or("")
                    .to_owned(),
            });
        }
    }
    if let Some(reason) = choice["finish_reason"].as_str() {
        out.push(Delta::Done {
            finish_reason: Some(reason.to_owned()),
            timings: v.get("timings").cloned(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_reasoning_tools_and_done() {
        let content = br#"data: {"choices":[{"delta":{"content":"Hej"},"finish_reason":null}]}"#;
        assert_eq!(
            parse_sse_line(content).unwrap(),
            vec![Delta::Content("Hej".into())]
        );

        let tool = br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"web_search","arguments":"{\"q\""}}]}}]}"#;
        assert_eq!(
            parse_sse_line(tool).unwrap(),
            vec![Delta::ToolCall {
                index: 0,
                id: Some("c1".into()),
                name: Some("web_search".into()),
                arguments: "{\"q\"".into()
            }]
        );

        let done =
            br#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":12}}"#;
        assert_eq!(
            parse_sse_line(done).unwrap(),
            vec![Delta::Done {
                finish_reason: Some("stop".into()),
                timings: Some(json!({"prompt_n":12}))
            }]
        );
    }

    #[test]
    fn ignores_non_data_and_done_marker() {
        assert!(parse_sse_line(b": keep-alive").unwrap().is_empty());
        assert!(parse_sse_line(b"data: [DONE]").unwrap().is_empty());
        assert!(parse_sse_line(b"").unwrap().is_empty());
    }
}
