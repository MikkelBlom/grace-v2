//! Runner protocol: a candidate model runs as a long-lived child process that speaks JSON
//! lines on stdin/stdout (stderr is free for its own logs). Keeping it alive means model load
//! is measured once and every request after that measures warm inference only, which is what
//! Grace will see in practice.
//!
//! 1. Runner prints `{"ready":true,"model":"…","load_ms":1234.5, …}` once loaded.
//! 2. Bench writes one request per line; runner answers each with one line echoing `id`.
//!    STT:  `{"id":"c1","audio":"C:/…/c1.wav"}`        → `{"id":"c1","text":"…","ms":120.5}`
//!    TTS:  `{"id":"s1","text":"…","out":"C:/…/s1.wav"}` → `{"id":"s1","ttfa_ms":90,"total_ms":400}`
//!    Error: `{"id":"…","error":"…"}`
//! 3. Bench closes stdin; runner exits.
//!
//! Runners can be written in any language. See `bench/runners/` for the Python ones.

use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Instant,
};

use anyhow::{Context, bail};
use serde_json::Value;

pub struct Runner {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    /// The runner's `ready` line, kept verbatim for the report.
    pub ready: Value,
    /// Wall time from spawn to `ready`, including interpreter start-up.
    pub startup_ms: f64,
}

impl Runner {
    pub fn spawn(argv: &[String]) -> anyhow::Result<Self> {
        let (program, args) = argv.split_first().context("empty runner command")?;
        let started = Instant::now();
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning runner {program}"))?;
        let stdin = BufWriter::new(child.stdin.take().context("runner stdin")?);
        let stdout = BufReader::new(child.stdout.take().context("runner stdout")?);
        let mut runner = Self {
            child,
            stdin,
            stdout,
            ready: Value::Null,
            startup_ms: 0.0,
        };
        let ready = runner.read_line()?;
        if ready.get("ready").and_then(Value::as_bool) != Some(true) {
            bail!("runner did not report ready: {ready}");
        }
        runner.ready = ready;
        runner.startup_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(runner)
    }

    /// Sends one request and returns the reply plus the round-trip wall time in ms.
    pub fn call(&mut self, request: &Value) -> anyhow::Result<(Value, f64)> {
        let started = Instant::now();
        writeln!(self.stdin, "{request}")?;
        self.stdin.flush()?;
        let reply = self.read_line()?;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        if reply.get("id") != request.get("id") {
            bail!("runner answered {reply} to request {request}");
        }
        Ok((reply, wall_ms))
    }

    fn read_line(&mut self) -> anyhow::Result<Value> {
        // Skip blank or non-JSON lines: some libraries print banners to stdout on import.
        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                bail!("runner exited before replying");
            }
            if let Ok(value) = serde_json::from_str::<Value>(line.trim()) {
                return Ok(value);
            }
            eprintln!("[runner stdout] {}", line.trim_end());
        }
    }

    pub fn finish(mut self) -> anyhow::Result<()> {
        drop(self.stdin);
        self.child.wait()?;
        Ok(())
    }
}
