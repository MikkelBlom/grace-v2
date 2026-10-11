//! Grace bake-off harness. Measures candidate models on this machine, on our own fixtures,
//! instead of trusting leaderboards.
//!
//!   grace-bench stt --manifest fixtures/stt/manifest.jsonl --label edda -- py -3.12 bench/runners/stt_hf_whisper.py --model …
//!   grace-bench tts --sentences fixtures/tts/sentences.da.txt --label piper -- py -3.12 bench/runners/tts_piper.py --voice …
//!   grace-bench listen bench-results/tts
//!
//! Each run writes `<out>/<kind>/<label>/report.json` and appends one summary line to
//! `<out>/summary.jsonl`, so results accumulate across sessions and machines.

mod listen;
mod llm;
mod metrics;
mod runner;

use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use metrics::{ErrorCount, char_errors, percentile, word_errors};
use runner::Runner;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Speech-to-text: accuracy (WER/CER) and warm latency per clip.
    Stt {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        label: String,
        #[arg(long, default_value = "bench-results")]
        out: PathBuf,
        /// Only run clips carrying this tag (e.g. `conversation`).
        #[arg(long)]
        tag: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
        /// Runner command line, after `--`.
        #[arg(last = true, required = true)]
        runner: Vec<String>,
    },
    /// Text-to-speech: time-to-first-audio and real-time factor per sentence.
    Tts {
        /// One sentence per line; blank lines and `#` comments are skipped.
        #[arg(long)]
        sentences: PathBuf,
        #[arg(long)]
        label: String,
        #[arg(long, default_value = "bench-results")]
        out: PathBuf,
        #[arg(last = true, required = true)]
        runner: Vec<String>,
    },
    /// Builds a blind A/B listening page from every TTS label under `dir`.
    Listen { dir: PathBuf },
    /// LLM latency and tool-choice accuracy against a running llama-server.
    Llm {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// Directory with `system.md`, `tools.json` and `prompts.jsonl`.
        #[arg(long, default_value = "fixtures/llm")]
        fixtures: PathBuf,
        #[arg(long)]
        label: String,
        #[arg(long, default_value = "bench-results")]
        out: PathBuf,
        #[arg(long, default_value_t = 2)]
        repeat: usize,
    },
}

/// One line of an STT manifest. `audio` is relative to the manifest file.
#[derive(Debug, Deserialize)]
struct Clip {
    id: String,
    audio: PathBuf,
    text: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Serialize)]
struct SttItem {
    id: String,
    tags: Vec<String>,
    audio_s: f64,
    reference: String,
    hypothesis: String,
    wer: f64,
    cer: f64,
    runner_ms: Option<f64>,
    wall_ms: f64,
    /// Whisper's own confidence, when the runner reports it (for hallucination filters).
    avg_logprob: Option<f64>,
    no_speech_prob: Option<f64>,
    error: Option<String>,
}

#[derive(Serialize)]
struct TtsItem {
    id: String,
    text: String,
    audio: String,
    audio_s: f64,
    ttfa_ms: Option<f64>,
    total_ms: Option<f64>,
    wall_ms: f64,
    rtf: Option<f64>,
    error: Option<String>,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Cmd::Stt {
            manifest,
            label,
            out,
            tag,
            limit,
            runner,
        } => run_stt(&manifest, &label, &out, tag.as_deref(), limit, &runner),
        Cmd::Tts {
            sentences,
            label,
            out,
            runner,
        } => run_tts(&sentences, &label, &out, &runner),
        Cmd::Llm {
            url,
            fixtures,
            label,
            out,
            repeat,
        } => llm::run(llm::Options {
            url: &url,
            dir: &fixtures,
            label: &label,
            out: &out,
            repeat,
        }),
        Cmd::Listen { dir } => {
            let page = listen::build(&dir)?;
            println!("wrote {}", page.display());
            Ok(())
        }
    }
}

fn run_stt(
    manifest: &Path,
    label: &str,
    out: &Path,
    tag: Option<&str>,
    limit: Option<usize>,
    argv: &[String],
) -> anyhow::Result<()> {
    let base = manifest.parent().unwrap_or(Path::new("."));
    let mut clips = Vec::new();
    for line in BufReader::new(File::open(manifest).context("opening manifest")?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let clip: Clip =
            serde_json::from_str(&line).with_context(|| format!("bad manifest line: {line}"))?;
        if tag.is_none_or(|t| clip.tags.iter().any(|c| c == t)) {
            clips.push(clip);
        }
    }
    clips.truncate(limit.unwrap_or(usize::MAX));
    anyhow::ensure!(!clips.is_empty(), "no clips selected");

    let mut runner = Runner::spawn(argv)?;
    eprintln!(
        "[{label}] ready in {:.0} ms: {}",
        runner.startup_ms, runner.ready
    );

    let mut items = Vec::new();
    let (mut words, mut chars) = (ErrorCount::default(), ErrorCount::default());
    let mut audio_total = 0.0;
    let (mut silent_clips, mut hallucinated) = (0usize, 0usize);
    for (i, clip) in clips.iter().enumerate() {
        let audio = absolute(&base.join(&clip.audio))?;
        let audio_s = wav_seconds(&audio).unwrap_or(0.0);
        let request = json!({ "id": clip.id, "audio": audio });
        let (reply, wall_ms) = runner.call(&request)?;
        let error = reply
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let hypothesis = reply
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let (w, c) = (
            word_errors(&clip.text, &hypothesis),
            char_errors(&clip.text, &hypothesis),
        );
        // Clips without words (noise, echo, media) measure hallucination, not WER.
        if clip.text.trim().is_empty() {
            silent_clips += 1;
            if !hypothesis.trim().is_empty() {
                hallucinated += 1;
            }
        } else {
            words = words.add(w);
            chars = chars.add(c);
        }
        audio_total += audio_s;
        eprintln!(
            "[{label}] {}/{} {} wer={:.2} {:.0} ms",
            i + 1,
            clips.len(),
            clip.id,
            w.rate(),
            wall_ms
        );
        items.push(SttItem {
            id: clip.id.clone(),
            tags: clip.tags.clone(),
            audio_s,
            reference: clip.text.clone(),
            hypothesis,
            wer: w.rate(),
            cer: c.rate(),
            runner_ms: reply.get("ms").and_then(Value::as_f64),
            wall_ms,
            avg_logprob: reply.get("avg_logprob").and_then(Value::as_f64),
            no_speech_prob: reply.get("no_speech_prob").and_then(Value::as_f64),
            error,
        });
    }
    let ready = runner.ready.clone();
    let startup_ms = runner.startup_ms;
    runner.finish()?;

    let latencies: Vec<f64> = items.iter().map(|i| i.wall_ms).collect();
    let infer_total: f64 = latencies.iter().sum();
    let summary = json!({
        "kind": "stt",
        "label": label,
        "at": unix_secs(),
        "clips": items.len(),
        "errors": items.iter().filter(|i| i.error.is_some()).count(),
        "wer": words.rate(),
        "cer": chars.rate(),
        "silent_clips": silent_clips,
        "hallucinated": hallucinated,
        "p50_ms": percentile(&latencies, 50.0),
        "p95_ms": percentile(&latencies, 95.0),
        "rtf": if audio_total > 0.0 { infer_total / 1000.0 / audio_total } else { 0.0 },
        "startup_ms": startup_ms,
        "runner": ready,
        "manifest": manifest,
        "tag": tag,
    });
    write_report(out, "stt", label, &summary, &items)
}

fn run_tts(sentences: &Path, label: &str, out: &Path, argv: &[String]) -> anyhow::Result<()> {
    let texts: Vec<String> = fs::read_to_string(sentences)
        .context("reading sentences")?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect();
    anyhow::ensure!(!texts.is_empty(), "no sentences");
    let audio_dir = out.join("tts").join(label);
    fs::create_dir_all(&audio_dir)?;
    let audio_dir = absolute(&audio_dir)?;

    let mut runner = Runner::spawn(argv)?;
    eprintln!(
        "[{label}] ready in {:.0} ms: {}",
        runner.startup_ms, runner.ready
    );

    let mut items = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let id = format!("s{:02}", i + 1);
        let wav = audio_dir.join(format!("{id}.wav"));
        let (reply, wall_ms) = runner.call(&json!({ "id": id, "text": text, "out": wav }))?;
        let error = reply
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let audio_s = wav_seconds(&wav).unwrap_or(0.0);
        let total_ms = reply.get("total_ms").and_then(Value::as_f64);
        let ttfa_ms = reply.get("ttfa_ms").and_then(Value::as_f64);
        eprintln!(
            "[{label}] {}/{} ttfa={} total={:.0} ms audio={audio_s:.1}s",
            i + 1,
            texts.len(),
            ttfa_ms.map_or("n/a".into(), |v| format!("{v:.0} ms")),
            total_ms.unwrap_or(wall_ms)
        );
        items.push(TtsItem {
            id,
            text: text.clone(),
            audio: format!(
                "{label}/{}",
                wav.file_name().unwrap_or_default().to_string_lossy()
            ),
            audio_s,
            ttfa_ms,
            total_ms,
            wall_ms,
            rtf: (audio_s > 0.0).then(|| total_ms.unwrap_or(wall_ms) / 1000.0 / audio_s),
            error,
        });
    }
    let ready = runner.ready.clone();
    let startup_ms = runner.startup_ms;
    runner.finish()?;

    let ttfa: Vec<f64> = items.iter().filter_map(|i| i.ttfa_ms).collect();
    let rtf: Vec<f64> = items.iter().filter_map(|i| i.rtf).collect();
    let summary = json!({
        "kind": "tts",
        "label": label,
        "at": unix_secs(),
        "sentences": items.len(),
        "errors": items.iter().filter(|i| i.error.is_some()).count(),
        "ttfa_p50_ms": percentile(&ttfa, 50.0),
        "ttfa_p95_ms": percentile(&ttfa, 95.0),
        "rtf_p50": percentile(&rtf, 50.0),
        "startup_ms": startup_ms,
        "runner": ready,
    });
    write_report(out, "tts", label, &summary, &items)
}

fn write_report<T: Serialize>(
    out: &Path,
    kind: &str,
    label: &str,
    summary: &Value,
    items: &[T],
) -> anyhow::Result<()> {
    let dir = out.join(kind).join(label);
    fs::create_dir_all(&dir)?;
    let report = json!({ "summary": summary, "items": items });
    fs::write(
        dir.join("report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("summary.jsonl"))?;
    writeln!(log, "{summary}")?;
    println!("{}", serde_json::to_string_pretty(summary)?);
    Ok(())
}

fn wav_seconds(path: &Path) -> Option<f64> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    Some(reader.duration() as f64 / spec.sample_rate as f64)
}

fn absolute(path: &Path) -> anyhow::Result<PathBuf> {
    Ok(std::path::absolute(path)?)
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
