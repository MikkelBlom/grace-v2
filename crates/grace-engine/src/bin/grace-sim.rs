//! `grace-sim`: runs Grace's real frontend (AEC3 → VAD → gate) on recorded or synthetic scenes,
//! faster than real time, and writes what STT would hear plus listening metrics.
//!
//!   grace-sim --scenes fixtures/corpora/scenes/far --out bench-results/sim/far [--no-ns]
//!
//! A scene is `<id>.mic.wav` (what the mic hears, 48 kHz mono) and `<id>.loop.wav` (what the
//! speakers played, 48 kHz mono), described by a line in `scenes.jsonl`:
//!   {"id": "...", "kind": "speech|echo|media|noise", "text": "...", "user": [[t0, t1]],
//!    "delay_ms": 120, "eval_from": 6.0}
//! `user` is when the person speaks (seconds from scene start); `text` what they say. Before
//! `eval_from` Grace talks alone so AEC3 has converged (live, it has been running all along);
//! speech detected there is reported as `warmup_triggers` (the cold-start case).
//!
//! Output: `summary.json` (detection, false triggers, timing), and bench manifests of the turn
//! audio for each signal (`aec`, `lin`, `raw`) cut at the VAD's edges (`vad-*.jsonl`) and at the
//! true edges (`oracle-*.jsonl`), for `grace-bench stt`.

use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use clap::Parser;
use grace::{
    audio_io::{Chunk, RATE, Source},
    frontend::{FrontEvent, Frontend, FrontendConfig, Produced, Shared},
    speech::write_wav,
};
use grace_core::vad_gate::VadGateConfig;
use serde::Deserialize;
use serde_json::json;

#[derive(Parser)]
struct Args {
    /// Directory with scenes.jsonl and the scene WAVs.
    #[arg(long)]
    scenes: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Turn AEC3's noise suppressor off (it is on in the live engine).
    #[arg(long)]
    no_ns: bool,
    /// Speaker delay the frontend starts from; defaults to the scene's true delay (calibrated).
    #[arg(long)]
    delay_ms: Option<f64>,
    #[arg(long)]
    limit: Option<usize>,
    /// VAD gate: start threshold added while Grace talks (default from VadGateConfig).
    #[arg(long)]
    talking_boost: Option<f32>,
    /// VAD gate: frames of speech needed to start while Grace talks.
    #[arg(long)]
    talking_min_frames: Option<u32>,
    /// Run VAD on AEC3's linear output instead of the full chain.
    #[arg(long)]
    vad_linear: bool,
    /// Double-talk detector threshold in dB (implies --vad-linear).
    #[arg(long)]
    double_talk_db: Option<f32>,
}

impl Args {
    fn gate(&self) -> VadGateConfig {
        let mut g = VadGateConfig::default();
        if let Some(b) = self.talking_boost {
            g.talking_boost = b;
        }
        if let Some(f) = self.talking_min_frames {
            g.talking_min_speech_frames = f;
        }
        g
    }
}

#[derive(Deserialize)]
struct Scene {
    id: String,
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    user: Vec<[f64; 2]>,
    #[serde(default = "default_delay")]
    delay_ms: f64,
    /// Scoring starts here; before it Grace talks so the echo canceller converges, as live.
    #[serde(default)]
    eval_from: f64,
}

fn default_delay() -> f64 {
    120.0
}

/// Scene time 0 on the frontend's clock (any positive offset works; QPC times are large too).
const T0: f64 = 1000.0;
const CHUNK: usize = (RATE / 100) as usize;
/// Same padding the conductor uses around a turn for STT.
const PAD_BEFORE: f64 = 0.3;
const PAD_AFTER: f64 = 0.15;

fn read_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut r =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = r.spec();
    anyhow::ensure!(
        spec.sample_rate == RATE && spec.channels == 1,
        "{}: need 48 kHz mono",
        path.display()
    );
    Ok(match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    })
}

struct Outcome {
    /// (start, stop) speech segments in scene seconds.
    segments: Vec<(f64, f64)>,
    shared: Shared,
    secs: f64,
}

fn run_scene(args: &Args, dir: &Path, scene: &Scene) -> anyhow::Result<Outcome> {
    let mic = read_wav(&dir.join(format!("{}.mic.wav", scene.id)))?;
    let lp = read_wav(&dir.join(format!("{}.loop.wav", scene.id)))?;
    let secs = mic.len() as f64 / RATE as f64;
    let mut front = Frontend::new(FrontendConfig {
        vad_model: PathBuf::from(".runtime/models/silero_vad_v6.2.3.onnx"),
        initial_delay_ms: args.delay_ms.unwrap_or(scene.delay_ms),
        lead_ms: 30.0,
        gate: args.gate(),
        noise_suppression: !args.no_ns,
        vad_on_linear: args.vad_linear || args.double_talk_db.is_some(),
        double_talk_db: args.double_talk_db,
    })?;
    let mut shared = Shared::new(secs + 2.0);
    let mut out = Produced::default();
    let (mut segments, mut open) = (Vec::new(), None::<f64>);
    let mut last_audible = f64::NEG_INFINITY;
    for (i, m) in mic.chunks(CHUNK).enumerate() {
        let time = T0 + (i * CHUNK) as f64 / RATE as f64;
        let l = lp
            .get(i * CHUNK..((i + 1) * CHUNK).min(lp.len()))
            .unwrap_or(&[]);
        // Grace counts as talking while the speakers are audible, plus the render thread's tail.
        if l.iter().any(|s| s.abs() > 0.002) {
            last_audible = time;
        }
        let talking = time - last_audible < 0.5;
        let mut lchunk = l.to_vec();
        lchunk.resize(m.len(), 0.0);
        for (source, samples) in [(Source::Loopback, lchunk), (Source::Mic, m.to_vec())] {
            front.push(
                &Chunk {
                    source,
                    time,
                    samples,
                },
                talking,
                &mut out,
            )?;
        }
        shared.absorb(&mut out);
        for ev in out.events.drain(..) {
            match ev {
                FrontEvent::SpeechStarted { time } => open = Some(time - T0),
                FrontEvent::SpeechStopped { time } => {
                    if let Some(s) = open.take() {
                        segments.push((s, time - T0));
                    }
                }
                FrontEvent::Delay { .. } => {}
            }
        }
    }
    if let Some(s) = open {
        segments.push((s, secs));
    }
    Ok(Outcome {
        segments,
        shared,
        secs,
    })
}

struct Manifests {
    files: Vec<(String, File)>,
}

impl Manifests {
    fn new(out: &Path) -> anyhow::Result<Self> {
        let mut files = Vec::new();
        for cut in ["vad", "oracle"] {
            for sig in ["aec", "lin", "raw"] {
                let name = format!("{cut}-{sig}");
                files.push((
                    name.clone(),
                    File::create(out.join(format!("{name}.jsonl")))?,
                ));
            }
        }
        Ok(Self { files })
    }

    fn add(
        &mut self,
        out: &Path,
        cut: &str,
        scene: &Scene,
        shared: &Shared,
        span: (f64, f64),
    ) -> anyhow::Result<()> {
        let (t0, t1) = (T0 + span.0 - PAD_BEFORE, T0 + span.1 + PAD_AFTER);
        for (sig, ring) in [
            ("aec", &shared.aec16),
            ("lin", &shared.lin16),
            ("raw", &shared.raw16),
        ] {
            let rel = format!("audio/{}.{cut}.{sig}.wav", scene.id);
            write_wav(&out.join(&rel), &ring.read_span(t0, t1), 16_000)?;
            let name = format!("{cut}-{sig}");
            let file = &mut self
                .files
                .iter_mut()
                .find(|(n, _)| *n == name)
                .expect("manifest")
                .1;
            writeln!(
                file,
                "{}",
                json!({"id": scene.id, "audio": rel, "text": scene.text, "tags": [scene.kind]})
            )?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Kind {
    scenes: usize,
    secs: f64,
    detected: usize,
    triggers: usize,
    false_speech_s: f64,
    onset: Vec<f64>,
    offset: Vec<f64>,
    split: usize,
    warmup_triggers: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let started = Instant::now();
    std::fs::create_dir_all(args.out.join("audio"))?;
    let mut scenes = Vec::new();
    for line in BufReader::new(File::open(args.scenes.join("scenes.jsonl"))?).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            scenes.push(serde_json::from_str::<Scene>(&line)?);
        }
    }
    scenes.truncate(args.limit.unwrap_or(usize::MAX));
    let mut manifests = Manifests::new(&args.out)?;
    let mut kinds: std::collections::BTreeMap<String, Kind> = Default::default();
    let mut audio_s = 0.0;
    let mut rows = File::create(args.out.join("scenes.jsonl"))?;
    for (i, scene) in scenes.iter().enumerate() {
        let mut o = run_scene(&args, &args.scenes, scene)?;
        audio_s += o.secs;
        let k = kinds.entry(scene.kind.clone()).or_default();
        k.scenes += 1;
        k.secs += o.secs - scene.eval_from;
        let before = o.segments.len();
        o.segments.retain(|(_, e)| *e > scene.eval_from);
        k.warmup_triggers += before - o.segments.len();
        writeln!(
            rows,
            "{}",
            json!({"id": scene.id, "kind": scene.kind, "user": scene.user, "segments": o.segments})
        )?;
        if let (Some(first), Some(last)) = (scene.user.first(), scene.user.last()) {
            let (u0, u1) = (first[0], last[1]);
            let inside: Vec<_> = o
                .segments
                .iter()
                .filter(|(s, e)| *e > u0 - 0.3 && *s < u1 + 0.3)
                .copied()
                .collect();
            // Speech outside the person's words: triggered by echo or noise.
            for (s, e) in o
                .segments
                .iter()
                .filter(|(s, e)| *e <= u0 - 0.3 || *s >= u1 + 0.3)
            {
                k.triggers += 1;
                k.false_speech_s += e - s;
            }
            if let (Some(a), Some(b)) = (inside.first(), inside.last()) {
                k.detected += 1;
                k.onset.push(a.0 - u0);
                k.offset.push(b.1 - u1);
                // A gap longer than the turn machine's longest wait splits the turn in two.
                k.split += inside
                    .windows(2)
                    .filter(|w| w[1].0 - w[0].1 > 1.6)
                    .count()
                    .min(1);
                manifests.add(&args.out, "vad", scene, &o.shared, (a.0, b.1))?;
            }
            manifests.add(&args.out, "oracle", scene, &o.shared, (u0, u1))?;
        } else {
            for (s, e) in &o.segments {
                k.triggers += 1;
                k.false_speech_s += e - s;
            }
            // What STT would be handed on a false trigger: must come back empty.
            if let (Some(a), Some(b)) = (o.segments.first(), o.segments.last()) {
                manifests.add(&args.out, "vad", scene, &o.shared, (a.0, b.1))?;
            }
        }
        if (i + 1) % 25 == 0 {
            eprintln!("{}/{} scenes", i + 1, scenes.len());
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    let mut summary = serde_json::Map::new();
    for (name, k) in &mut kinds {
        summary.insert(
            name.clone(),
            json!({
                "scenes": k.scenes,
                "hours": k.secs / 3600.0,
                "detected": k.detected,
                "triggers": k.triggers,
                "triggers_per_hour": k.triggers as f64 / (k.secs / 3600.0),
                "false_speech_s": k.false_speech_s,
                "split_turns": k.split,
                "warmup_triggers": k.warmup_triggers,
                "onset_median_s": median(&mut k.onset),
                "offset_median_s": median(&mut k.offset),
            }),
        );
    }
    let wall = started.elapsed().as_secs_f64();
    let report = json!({
        "scenes": args.scenes,
        "noise_suppression": !args.no_ns,
        "delay_ms": args.delay_ms,
        "gate": format!("{:?}", args.gate()),
        "vad_linear": args.vad_linear,
        "double_talk_db": args.double_talk_db,
        "audio_s": audio_s,
        "wall_s": wall,
        "speedup": audio_s / wall,
        "kinds": summary,
    });
    std::fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
