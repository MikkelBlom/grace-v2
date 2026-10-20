//! `grace-turn-sim`: the turn-taking policy on recorded turns, in virtual time.
//!
//!   grace-turn-sim --scenes fixtures/corpora/scenes/turns --out bench-results/turn-sim/recovery
//!   grace-turn-sim ... --policy baseline          (the policy of live runs 1-4)
//!   grace-turn-sim ... --join-check               (re-transcribe vs join transcripts)
//!
//! Scenes come from `bench/sim/make_turns.py`: one person's turn, often paused mid-way and
//! resumed. Each scene runs through Grace's real frontend (AEC3 → Silero VAD → gate), then the
//! real [`TurnMachine`] is driven in virtual time by a model of the conductor: Smart Turn (da1)
//! and STT (edda via the speech server, which must be running) give real scores and words for
//! every pause, and their measured compute times delay the results. The LLM + TTS path is a
//! constant (`--reply-ms`, live run 4 medians: memory recall 94 ms + commit → first audio 522 ms); the
//! speakers and Bluetooth add their own ~150 ms on top live. Model results are cached per audio
//! span (`--cache`), so policy sweeps re-run in seconds.
//!
//! The person doesn't react to Grace (a headset: none of her voice reaches the mic), which is
//! the hard case: they go on talking whatever she does.
//!
//! Per scene: did Grace jump in before the person was done (`early`), was it heard, was it
//! recovered by merging, did the final reply answer the whole turn, and how long from the true
//! end of speech to her first audio.

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::Context;
use clap::Parser;
use grace::{
    audio_io::{Chunk, RATE, Source},
    frontend::{FrontEvent, Frontend, FrontendConfig, Produced, Shared},
    speech::Stt,
};
use grace_core::{
    turn::{Command, Config as TurnConfig, Input, TurnMachine},
    vad_gate::VadGateConfig,
};
use grace_vad::smart_turn::SmartTurn;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Parser)]
struct Args {
    /// Directory with scenes.jsonl and `<id>.mic.wav` (bench/sim/make_turns.py).
    #[arg(long)]
    scenes: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// `baseline` (live runs 1-4) or `recovery` (merging + speculative replies, the default).
    #[arg(long, default_value = "recovery")]
    policy: String,
    #[arg(long)]
    threshold: Option<f32>,
    #[arg(long)]
    unfinished: Option<f32>,
    #[arg(long)]
    silence_ms: Option<u64>,
    #[arg(long)]
    silence_unfinished_ms: Option<u64>,
    /// Merge window in ms; 0 turns merging off.
    #[arg(long)]
    merge_ms: Option<u64>,
    #[arg(long)]
    speculate: Option<bool>,
    #[arg(long)]
    min_commit_ms: Option<u64>,
    /// VAD gate: quiet frames (32 ms) before a pause is reported (default 4).
    #[arg(long)]
    hangover_frames: Option<u32>,
    /// Commit → first audio when nothing was prepared (LLM first sentence + TTS first audio).
    #[arg(long, default_value_t = 620.0)]
    reply_ms: f64,
    /// How long each reply plays if nobody stops it.
    #[arg(long, default_value_t = 6.0)]
    reply_secs: f64,
    #[arg(long, default_value = "bench-results/turn-sim/cache.jsonl")]
    cache: PathBuf,
    #[arg(long, default_value = "127.0.0.1:7810")]
    stt: String,
    #[arg(long)]
    limit: Option<usize>,
    /// Instead of the policy run: WER of transcribing a paused turn whole vs part by part.
    #[arg(long)]
    join_check: bool,
}

impl Args {
    fn gate(&self) -> VadGateConfig {
        let mut g = VadGateConfig::default();
        if let Some(v) = self.hangover_frames {
            g.hangover_frames = v;
        }
        g
    }

    fn turn_config(&self) -> anyhow::Result<TurnConfig> {
        let mut c = match self.policy.as_str() {
            "baseline" => TurnConfig::without_recovery(),
            "recovery" => TurnConfig::default(),
            p => anyhow::bail!("unknown policy {p}"),
        };
        if let Some(v) = self.threshold {
            c.turn_end_threshold = v;
        }
        if let Some(v) = self.unfinished {
            c.unfinished_threshold = v;
        }
        if let Some(v) = self.silence_ms {
            c.max_trailing_silence = Duration::from_millis(v);
        }
        if let Some(v) = self.silence_unfinished_ms {
            c.max_trailing_silence_unfinished = Duration::from_millis(v);
        }
        if let Some(v) = self.merge_ms {
            c.merge_window = (v > 0).then(|| Duration::from_millis(v));
        }
        if let Some(v) = self.min_commit_ms {
            c.min_commit_silence = Duration::from_millis(v);
        }
        if let Some(v) = self.speculate {
            c.speculate = v;
        }
        Ok(c)
    }
}

#[derive(Deserialize)]
struct Scene {
    id: String,
    kind: String,
    texts: Vec<String>,
    parts: Vec<[f64; 2]>,
    true_end: f64,
}

const T0: f64 = 1000.0;
const CHUNK: usize = (RATE / 100) as usize;
const STEP: f64 = 0.01;
/// The conductor's tick.
const TICK: f64 = 0.03;

fn read_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut r =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    anyhow::ensure!(r.spec().sample_rate == RATE && r.spec().channels == 1);
    Ok(match r.spec().sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u32 << (r.spec().bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    })
}

/// The frontend over the whole scene: speech edges (emitted at, back-dated to) and the rings.
type FrontPass = (Vec<(f64, FrontEvent)>, Shared, f64);

fn frontend(dir: &Path, scene: &Scene, gate: VadGateConfig) -> anyhow::Result<FrontPass> {
    let mic = read_wav(&dir.join(format!("{}.mic.wav", scene.id)))?;
    let secs = mic.len() as f64 / RATE as f64;
    let mut front = Frontend::new(FrontendConfig {
        vad_model: PathBuf::from(".runtime/models/silero_vad_v6.2.3.onnx"),
        initial_delay_ms: 120.0,
        lead_ms: 30.0,
        gate,
        noise_suppression: true,
        vad_on_linear: false,
        double_talk_db: None,
    })?;
    let mut shared = Shared::new(secs + 2.0);
    let mut out = Produced::default();
    let mut events = Vec::new();
    for (i, m) in mic.chunks(CHUNK).enumerate() {
        let time = T0 + (i * CHUNK) as f64 / RATE as f64;
        for (source, samples) in [
            (Source::Loopback, vec![0.0; m.len()]),
            (Source::Mic, m.to_vec()),
        ] {
            front.push(
                &Chunk {
                    source,
                    time,
                    samples,
                },
                false,
                &mut out,
            )?;
        }
        shared.absorb(&mut out);
        let emitted = time + m.len() as f64 / RATE as f64 - T0;
        for ev in out.events.drain(..) {
            if !matches!(ev, FrontEvent::Delay { .. }) {
                events.push((emitted, ev));
            }
        }
    }
    Ok((events, shared, secs))
}

/// Smart Turn and STT results per audio span, kept across runs.
struct Models {
    smart_turn: SmartTurn,
    stt: Option<Stt>,
    stt_addr: String,
    cache: HashMap<String, Value>,
    file: File,
}

impl Models {
    fn open(path: &Path, stt_addr: &str) -> anyhow::Result<Self> {
        let mut cache = HashMap::new();
        if let Ok(f) = File::open(path) {
            for line in BufReader::new(f).lines() {
                let v: Value = serde_json::from_str(&line?)?;
                cache.insert(v["key"].as_str().unwrap_or_default().to_owned(), v);
            }
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            smart_turn: SmartTurn::new(Path::new(".runtime/models/smart-turn-v3.2-da1.onnx"))?,
            stt: None,
            stt_addr: stt_addr.to_owned(),
            cache,
            file: std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?,
        })
    }

    fn put(&mut self, key: String, mut v: Value) -> anyhow::Result<Value> {
        v["key"] = json!(key);
        writeln!(self.file, "{v}")?;
        self.cache.insert(key, v.clone());
        Ok(v)
    }

    /// Turn-end probability and compute ms, read exactly as the conductor reads it.
    fn score(
        &mut self,
        id: &str,
        sh: &Shared,
        start: f64,
        stop: f64,
    ) -> anyhow::Result<(f32, f64)> {
        let key = format!("st|{id}|{:.0}|{:.0}", start * 1e3, stop * 1e3);
        let v = match self.cache.get(&key) {
            Some(v) => v.clone(),
            None => {
                let audio = sh.aec16.read_span(T0 + start - 0.5, T0 + stop + 0.2);
                let t = Instant::now();
                let p = self.smart_turn.predict(&audio)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                self.put(key, json!({"p": p, "ms": ms}))?
            }
        };
        Ok((
            v["p"].as_f64().unwrap_or(0.0) as f32,
            v["ms"].as_f64().unwrap_or(40.0),
        ))
    }

    /// STT text (empty if the ekko check calls it a hallucination) and server ms.
    async fn transcribe(
        &mut self,
        id: &str,
        sh: &Shared,
        start: f64,
        stop: f64,
    ) -> anyhow::Result<(String, f64)> {
        let key = format!("stt|{id}|{:.0}|{:.0}", start * 1e3, stop * 1e3);
        let v = match self.cache.get(&key) {
            Some(v) => v.clone(),
            None => {
                if self.stt.is_none() {
                    self.stt = Some(Stt::connect(&self.stt_addr).await?);
                }
                let audio = sh.lin16.read_span(T0 + start - 0.3, T0 + stop + 0.15);
                let t = self
                    .stt
                    .as_ref()
                    .expect("connected")
                    .transcribe(&audio)
                    .await?;
                let text = if t.looks_hallucinated() {
                    String::new()
                } else {
                    t.text
                };
                self.put(key, json!({"text": text, "ms": t.ms}))?
            }
        };
        Ok((
            v["text"].as_str().unwrap_or_default().to_owned(),
            v["ms"].as_f64().unwrap_or(250.0),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum End {
    Playing,
    Finished,
    /// Interrupted by a new turn: what played stays in the conversation.
    Cancelled,
    /// Retracted by a merge: as if never said.
    Retracted,
    /// A prepared reply nobody released.
    Discarded,
}

struct Reply {
    turn: u64,
    text: String,
    /// When its first audio exists (LLM + TTS done with the first sentence).
    ready: f64,
    released: Option<f64>,
    paused: bool,
    audible: Option<f64>,
    played: f64,
    end: End,
    respond_at: Option<f64>,
}

struct VTurn {
    start: f64,
    stop: f64,
    epoch: u32,
}

enum Due {
    Score { turn: u64, epoch: u32, p: f32 },
    Transcript { turn: u64, epoch: u32, text: String },
}

/// One scene, in virtual time, with the conductor modelled.
async fn simulate(
    args: &Args,
    cfg: &TurnConfig,
    scene: &Scene,
    events: &[(f64, FrontEvent)],
    sh: &Shared,
    secs: f64,
    models: &mut Models,
) -> anyhow::Result<Value> {
    let mut m = TurnMachine::new(cfg.clone());
    let mut turns: HashMap<u64, VTurn> = HashMap::new();
    let mut replies: Vec<Reply> = Vec::new();
    let mut due: Vec<(f64, Due)> = Vec::new();
    let (mut next_ev, mut last_start, mut next_tick) = (0, 0.0, 0.0);
    let mut log = Vec::new();
    let mut prepares = 0;
    let d = |t: f64| Duration::from_secs_f64(t.max(0.0));
    let mut t = 0.0;
    while t < secs {
        let mut inputs: Vec<(Input, f64)> = Vec::new();
        while next_ev < events.len() && events[next_ev].0 <= t {
            match events[next_ev].1 {
                FrontEvent::SpeechStarted { time } => {
                    last_start = time - T0;
                    inputs.push((Input::SpeechStarted, time - T0));
                }
                FrontEvent::SpeechStopped { time } => {
                    if let Some(v) = turns
                        .values_mut()
                        .max_by(|a, b| a.start.total_cmp(&b.start))
                    {
                        v.stop = time - T0;
                    }
                    inputs.push((Input::SpeechStopped, time - T0));
                }
                FrontEvent::Delay { .. } => {}
            }
            next_ev += 1;
        }
        let (now_due, later): (Vec<_>, Vec<_>) = due.drain(..).partition(|(at, _)| *at <= t);
        due = later;
        for (_, x) in now_due {
            match x {
                Due::Score { turn, epoch, p }
                    if turns.get(&turn).is_some_and(|v| v.epoch == epoch) =>
                {
                    inputs.push((
                        Input::TurnEndScore {
                            turn,
                            probability: p,
                        },
                        t,
                    ));
                }
                Due::Transcript { turn, epoch, text }
                    if turns.get(&turn).is_some_and(|v| v.epoch == epoch) =>
                {
                    inputs.push((Input::Transcript { turn, text }, t));
                }
                _ => {}
            }
        }
        // Playback of the released reply (at most one plays at a time).
        for r in replies.iter_mut().filter(|r| r.end == End::Playing) {
            if let Some(rel) = r.released
                && !r.paused
                && t >= rel.max(r.ready)
            {
                if r.audible.is_none() {
                    r.audible = Some(t);
                    inputs.push((Input::ReplyAudible { turn: r.turn }, t));
                }
                r.played += STEP;
                if r.played >= args.reply_secs {
                    r.end = End::Finished;
                    inputs.push((Input::ReplyFinished { turn: r.turn }, t));
                }
            }
        }
        if t >= next_tick {
            inputs.push((Input::Tick, t));
            next_tick += TICK;
        }
        for (input, at) in inputs {
            for cmd in m.step(input, d(at)) {
                log.push(json!([round(t), format!("{cmd:?}")]));
                match cmd {
                    Command::BeginTurn(id) => {
                        turns.insert(
                            id,
                            VTurn {
                                start: last_start,
                                stop: t,
                                epoch: 0,
                            },
                        );
                    }
                    Command::ReopenTurn(id) | Command::DiscardTranscript(id) => {
                        if let Some(v) = turns.get_mut(&id) {
                            v.epoch += 1;
                        }
                        for r in replies.iter_mut().filter(|r| {
                            r.turn == id && r.released.is_none() && r.end == End::Playing
                        }) {
                            r.end = End::Discarded;
                        }
                    }
                    Command::EvaluatePause(id) => {
                        let Some(v) = turns.get_mut(&id) else {
                            continue;
                        };
                        v.epoch += 1;
                        let (start, stop, epoch) = (v.start, v.stop, v.epoch);
                        let (p, st_ms) = models.score(&scene.id, sh, start, stop)?;
                        let (text, stt_ms) = models.transcribe(&scene.id, sh, start, stop).await?;
                        due.push((t + st_ms / 1e3, Due::Score { turn: id, epoch, p }));
                        due.push((
                            t + stt_ms / 1e3 + 0.01,
                            Due::Transcript {
                                turn: id,
                                epoch,
                                text,
                            },
                        ));
                    }
                    Command::Prepare { turn, text } => {
                        for r in replies.iter_mut().filter(|r| {
                            r.turn == turn && r.released.is_none() && r.end == End::Playing
                        }) {
                            r.end = End::Discarded;
                        }
                        prepares += 1;
                        replies.push(Reply {
                            turn,
                            text,
                            ready: t + args.reply_ms / 1e3,
                            released: None,
                            paused: false,
                            audible: None,
                            played: 0.0,
                            end: End::Playing,
                            respond_at: None,
                        });
                    }
                    Command::Respond { turn, text } => {
                        let prepared = replies.iter_mut().find(|r| {
                            r.turn == turn
                                && r.text == text
                                && r.released.is_none()
                                && r.end == End::Playing
                        });
                        match prepared {
                            Some(r) => {
                                r.released = Some(t);
                                r.respond_at = Some(t);
                            }
                            None => {
                                for r in replies
                                    .iter_mut()
                                    .filter(|r| r.turn == turn && r.released.is_none())
                                {
                                    r.end = End::Discarded;
                                }
                                replies.push(Reply {
                                    turn,
                                    text,
                                    ready: t + args.reply_ms / 1e3,
                                    released: Some(t),
                                    paused: false,
                                    audible: None,
                                    played: 0.0,
                                    end: End::Playing,
                                    respond_at: Some(t),
                                });
                            }
                        }
                    }
                    Command::DropTurn(id) => {
                        for r in replies
                            .iter_mut()
                            .filter(|r| r.turn == id && r.released.is_none())
                        {
                            r.end = End::Discarded;
                        }
                    }
                    Command::PauseReply(id) | Command::ResumeReply(id) => {
                        let pause = matches!(cmd, Command::PauseReply(_));
                        if let Some(r) = replies
                            .iter_mut()
                            .rev()
                            .find(|r| r.turn == id && r.released.is_some())
                        {
                            r.paused = pause;
                        }
                    }
                    Command::CancelReply(id) | Command::RetractReply(id) => {
                        let end = if matches!(cmd, Command::CancelReply(_)) {
                            End::Cancelled
                        } else {
                            End::Retracted
                        };
                        if let Some(r) = replies
                            .iter_mut()
                            .rev()
                            .find(|r| r.turn == id && r.released.is_some() && r.end == End::Playing)
                        {
                            r.end = end;
                        }
                    }
                    Command::DuckReply | Command::UnduckReply => {}
                }
            }
        }
        t += STEP;
    }

    // --- Outcome --------------------------------------------------------------------------
    let first_end = scene.parts[0][1];
    let last_start_true = scene.parts.last().expect("parts")[0];
    let end = scene.true_end;
    let released: Vec<&Reply> = replies.iter().filter(|r| r.released.is_some()).collect();
    let early = released
        .iter()
        .any(|r| r.respond_at.is_some_and(|a| a < end - 0.05));
    let heard_early: f64 = released
        .iter()
        .filter_map(|r| r.audible.map(|a| (a, r)))
        .filter(|(a, _)| *a < end)
        .map(|(a, r)| r.played.min(end - a).max(0.0))
        .sum();
    // Replies that stay in the conversation, in order.
    let kept: Vec<&&Reply> = released
        .iter()
        .filter(|r| r.audible.is_some() && r.end != End::Retracted)
        .collect();
    let final_reply = kept.last();
    let covers_whole = final_reply.is_some_and(|r| {
        let start = turns.get(&r.turn).map_or(f64::INFINITY, |v| v.start);
        // The turn began during his first part (VAD onsets run late) and was committed after
        // his last part began.
        start < first_end - 0.2 && r.respond_at.is_some_and(|a| a >= last_start_true)
    });
    let half_answers = kept
        .iter()
        .filter(|r| r.audible.is_some_and(|a| a < end))
        .count();
    let outcome = match final_reply {
        None => "unanswered",
        Some(_) if !covers_whole || half_answers > 0 => "split",
        Some(_) if !early => "clean",
        Some(_) if heard_early > 0.0 => "merged_heard",
        Some(_) => "merged_silent",
    };
    let latency = final_reply.and_then(|r| r.audible).map(|a| a - end);
    let discarded = replies.iter().filter(|r| r.end == End::Discarded).count();
    let retracted = replies.iter().filter(|r| r.end == End::Retracted).count();
    Ok(json!({
        "id": scene.id, "kind": scene.kind, "parts": scene.parts.len(),
        "gaps": scene.parts.windows(2).map(|w| round(w[1][0] - w[0][1])).collect::<Vec<_>>(),
        "outcome": outcome, "early": early, "heard_early_s": round(heard_early),
        "latency": latency.map(round), "llm_requests": replies.len(), "prepares": prepares,
        "discarded": discarded, "retracted": retracted, "half_answers": half_answers,
        "final_text": final_reply.map(|r| r.text.clone()),
        "reference": scene.texts.join(" "),
        "log": log,
    }))
}

fn round(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn pct(v: &mut [f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(round(v[((v.len() - 1) as f64 * q).round() as usize]))
}

fn summarize(rows: &[Value], kind: Option<&str>) -> Value {
    let rows: Vec<&Value> = rows
        .iter()
        .filter(|r| kind.is_none_or(|k| r["kind"] == k))
        .collect();
    let n = rows.len() as f64;
    let count = |o: &str| rows.iter().filter(|r| r["outcome"] == o).count();
    let share = |c: usize| round(c as f64 / n.max(1.0));
    let mut lat: Vec<f64> = rows.iter().filter_map(|r| r["latency"].as_f64()).collect();
    let under = lat.iter().filter(|l| **l < 0.8).count();
    let early = rows.iter().filter(|r| r["early"] == true).count();
    let sum = |k: &str| {
        rows.iter()
            .map(|r| r[k].as_f64().unwrap_or(0.0))
            .sum::<f64>()
    };
    json!({
        "scenes": rows.len(),
        "early_cut": share(early),
        "merged_silent": share(count("merged_silent")),
        "merged_heard": share(count("merged_heard")),
        "merge_rate_of_early": round((count("merged_silent") + count("merged_heard")) as f64 / (early as f64).max(1.0)),
        "split": share(count("split")),
        // Split, and Grace audibly answered part of the turn (her half-answer stays).
        "split_heard": share(rows.iter().filter(|r| r["outcome"] == "split" && r["half_answers"].as_u64().unwrap_or(0) > 0).count()),
        "unanswered": share(count("unanswered")),
        "clean": share(count("clean")),
        "latency_p50": pct(&mut lat, 0.5),
        "latency_p90": pct(&mut lat, 0.9),
        "under_800ms": share(under),
        "heard_early_s_mean": round(sum("heard_early_s") / n.max(1.0)),
        "llm_requests_per_scene": round(sum("llm_requests") / n.max(1.0)),
        "discarded_per_scene": round(sum("discarded") / n.max(1.0)),
    })
}

/// Word error rate (lowercase words, punctuation ignored).
fn wer(reference: &str, hyp: &str) -> (usize, usize) {
    let words = |s: &str| {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
    };
    let (r, h) = (words(reference), words(hyp));
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hw) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(rw != hw))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    (prev[h.len()], r.len())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = args.turn_config()?;
    std::fs::create_dir_all(&args.out)?;
    let mut scenes = Vec::new();
    for line in BufReader::new(File::open(args.scenes.join("scenes.jsonl"))?).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            scenes.push(serde_json::from_str::<Scene>(&line)?);
        }
    }
    scenes.truncate(args.limit.unwrap_or(usize::MAX));
    let mut models = Models::open(&args.cache, &args.stt)?;
    let started = Instant::now();

    if args.join_check {
        // Oracle edges, conductor padding: one transcript of the whole paused turn vs one per
        // part, joined with a space.
        let (mut whole, mut joined, mut words) = (0, 0, 0);
        let mut rows = File::create(args.out.join("join-check.jsonl"))?;
        for scene in scenes.iter().filter(|s| s.parts.len() > 1) {
            let (_, sh, _) = frontend(&args.scenes, scene, args.gate())?;
            let reference = scene.texts.join(" ");
            let first = scene.parts[0][0];
            let last = scene.parts.last().expect("parts")[1];
            let (w, _) = models.transcribe(&scene.id, &sh, first, last).await?;
            let mut pieces = Vec::new();
            for p in &scene.parts {
                pieces.push(models.transcribe(&scene.id, &sh, p[0], p[1]).await?.0);
            }
            let j = pieces.join(" ");
            let (ew, n) = wer(&reference, &w);
            let (ej, _) = wer(&reference, &j);
            whole += ew;
            joined += ej;
            words += n;
            writeln!(
                rows,
                "{}",
                json!({"id": scene.id, "reference": reference, "whole": w, "joined": j, "err_whole": ew, "err_joined": ej, "words": n})
            )?;
        }
        let summary = json!({
            "scenes": scenes.iter().filter(|s| s.parts.len() > 1).count(),
            "words": words,
            "wer_whole": round(whole as f64 / words.max(1) as f64),
            "wer_joined": round(joined as f64 / words.max(1) as f64),
        });
        std::fs::write(
            args.out.join("join-check.json"),
            serde_json::to_string_pretty(&summary)?,
        )?;
        println!("{summary}");
        return Ok(());
    }

    let mut rows = Vec::new();
    let mut file = File::create(args.out.join("scenes.jsonl"))?;
    for scene in &scenes {
        let (events, sh, secs) = frontend(&args.scenes, scene, args.gate())?;
        let row = simulate(&args, &cfg, scene, &events, &sh, secs, &mut models).await?;
        writeln!(file, "{row}")?;
        rows.push(row);
    }
    let summary = json!({
        "policy": args.policy,
        "config": format!("{cfg:?}"),
        "reply_ms": args.reply_ms,
        "hangover_frames": args.gate().hangover_frames,
        "all": summarize(&rows, None),
        "multi": summarize(&rows, Some("multi")),
        "single": summarize(&rows, Some("single")),
        "wall_s": round(started.elapsed().as_secs_f64()),
    });
    std::fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
