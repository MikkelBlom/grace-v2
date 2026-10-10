//! Measures how much of the PC's own playback the OS echo canceller removes (ADR 0006, option A).
//!
//!   cargo run --release -p grace-audio --example aec_test -- --play [--mic <name>] [--speakers <name>] [--clip <wav>]
//!
//! Without `--play` it only prints which devices and clip it would use: this tool makes noise.
//!
//! Plays a clip through the speakers while recording, in parallel:
//! - the microphone RAW (no voice processing): how much echo reaches the mic;
//! - the microphone in COMMUNICATIONS mode with AEC referenced to the speakers: what Grace hears;
//! - LOOPBACK of the speakers: proof the clip actually played.
//!
//! Keep the room quiet and don't talk during the run. Echo reduction is the drop in level from
//! raw to AEC during playback; it includes noise suppression, which is fine: that is what
//! Grace's VAD will see. All three recordings are written to `bench-results/aec/<time>/`.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use grace_audio::wasapi::{self, Capture, CaptureMode, Endpoint, Flow, Render};

const BASELINE: Duration = Duration::from_millis(1500);
const TAIL: Duration = Duration::from_millis(1000);

struct Take {
    name: &'static str,
    rate: u32,
    samples: Vec<f32>,
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn record(
    name: &'static str,
    ep: Endpoint,
    mode: CaptureMode,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<anyhow::Result<Take>> {
    thread::spawn(move || {
        wasapi::com_init()?;
        let mut cap =
            Capture::open(&ep, &mode).with_context(|| format!("opening {name} capture"))?;
        let mut samples = Vec::new();
        cap.start()?;
        while !stop.load(Ordering::Relaxed) {
            cap.read_mono(&mut samples)?;
            thread::sleep(Duration::from_millis(10));
        }
        cap.read_mono(&mut samples)?;
        Ok(Take {
            name,
            rate: cap.rate(),
            samples,
        })
    })
}

fn load_clip(path: Option<String>) -> anyhow::Result<(Vec<f32>, u32)> {
    let paths: Vec<PathBuf> = match path {
        Some(p) => vec![p.into()],
        None => {
            // Grace's own voice is the echo that matters most: use the English TTS bake-off takes.
            let dir = Path::new("bench-results/en/tts/kokoro-af_heart");
            let mut v: Vec<_> = std::fs::read_dir(dir)
                .with_context(|| format!("no --clip given and {} is missing", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "wav"))
                .collect();
            v.sort();
            v.truncate(6);
            v
        }
    };
    let mut out = Vec::new();
    let mut rate = 0;
    for p in paths {
        let mut reader =
            hound::WavReader::open(&p).with_context(|| format!("reading {}", p.display()))?;
        let spec = reader.spec();
        rate = spec.sample_rate;
        let ch = spec.channels as usize;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => {
                let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / scale))
                    .collect::<Result<_, _>>()?
            }
        };
        out.extend(
            samples
                .chunks_exact(ch)
                .map(|f| f.iter().sum::<f32>() / ch as f32),
        );
        out.extend(std::iter::repeat_n(0.0, rate as usize / 4));
    }
    Ok((out, rate))
}

fn dbfs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let rms =
        (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    20.0 * rms.max(1e-9).log10()
}

fn window(take: &Take, from: Duration, to: Duration) -> &[f32] {
    let a = (from.as_secs_f64() * take.rate as f64) as usize;
    let b = ((to.as_secs_f64() * take.rate as f64) as usize).min(take.samples.len());
    &take.samples[a.min(b)..b]
}

fn save(dir: &Path, take: &Take) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: take.rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(dir.join(format!("{}.wav", take.name)), spec)?;
    for &s in &take.samples {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    wasapi::com_init()?;
    let mic = match arg("--mic") {
        Some(n) => wasapi::find(Flow::Capture, &[&n])?,
        None => wasapi::default_endpoint(Flow::Capture, true)?,
    };
    let speakers = match arg("--speakers") {
        Some(n) => wasapi::find(Flow::Render, &[&n])?,
        None => wasapi::find(Flow::Render, &["speaker", "højttaler", "realtek"])?,
    };
    println!("mic:      {}\nspeakers: {}", mic.name, speakers.name);
    let (clip, clip_rate) = load_clip(arg("--clip"))?;
    println!("clip:     {:.1} s", clip.len() as f64 / clip_rate as f64);
    if !std::env::args().any(|a| a == "--play") {
        println!(
            "
Dry run. Add --play to play the clip through the speakers and measure."
        );
        return Ok(());
    }

    let stop = Arc::new(AtomicBool::new(false));
    let started = Instant::now();
    let takes = [
        record("raw", mic.clone(), CaptureMode::Raw, stop.clone()),
        record(
            "aec",
            mic.clone(),
            CaptureMode::Communications {
                aec_reference: Some(speakers.clone()),
            },
            stop.clone(),
        ),
        record(
            "loopback",
            speakers.clone(),
            CaptureMode::Loopback,
            stop.clone(),
        ),
    ];

    thread::sleep(BASELINE);
    let mut render = Render::open(&speakers)?;
    let pcm = wasapi::resample_linear(&clip, clip_rate, render.rate());
    let play_from = started.elapsed();
    println!(
        "playing {:.1} s ...",
        pcm.len() as f64 / render.rate() as f64
    );
    render.play_blocking(&pcm)?;
    let play_to = started.elapsed();
    thread::sleep(TAIL);
    stop.store(true, Ordering::Relaxed);

    let takes: Vec<Take> = takes
        .into_iter()
        .map(|h| h.join().expect("capture thread panicked"))
        .collect::<anyhow::Result<_>>()?;

    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let dir = PathBuf::from(format!("bench-results/aec/{stamp}"));
    std::fs::create_dir_all(&dir)?;
    println!(
        "\n{:<9} {:>12} {:>12}",
        "stream", "quiet dBFS", "playing dBFS"
    );
    let mut levels = std::collections::HashMap::new();
    for take in &takes {
        // Skip the first 300 ms of the baseline: streams settle after Start().
        let quiet = dbfs(window(take, Duration::from_millis(300), BASELINE));
        let playing = dbfs(window(take, play_from, play_to));
        println!("{:<9} {quiet:>12.1} {playing:>12.1}", take.name);
        levels.insert(take.name, (quiet, playing));
        save(&dir, take)?;
    }
    let (_, raw) = levels["raw"];
    let (aec_quiet, aec) = levels["aec"];
    println!(
        "\necho reduction (raw → aec, while playing): {:.1} dB",
        raw - aec
    );
    println!(
        "aec residual above its own quiet floor:    {:.1} dB",
        aec - aec_quiet
    );
    println!("recordings: {}", dir.display());
    Ok(())
}
