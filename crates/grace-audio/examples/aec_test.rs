//! Measures how much of the PC's own playback the OS echo canceller removes (ADR 0006, option A).
//!
//!   cargo run --release -p grace-audio --example aec_test -- --play [--mic <name>] [--speakers <name>] [--clip <wav>]
//!
//! `--analyse <dir>` re-analyses a saved run without playing anything.
//! Devices default to `config/audio.json`. Without `--play` it only prints which devices and
//! clip it would use: this tool makes noise.
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

/// Mean power per 10 ms frame.
fn envelope(take: &Take) -> Vec<f64> {
    let frame = take.rate as usize / 100;
    take.samples
        .chunks_exact(frame)
        .map(|c| c.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / frame as f64)
        .collect()
}

fn to_db(power: f64) -> f64 {
    10.0 * power.max(1e-12).log10()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let la: Vec<f64> = a.iter().map(|&x| to_db(x)).collect();
    let lb: Vec<f64> = b.iter().map(|&x| to_db(x)).collect();
    let (ma, mb) = (
        la.iter().sum::<f64>() / la.len() as f64,
        lb.iter().sum::<f64>() / lb.len() as f64,
    );
    let cov: f64 = la.iter().zip(&lb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = la.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = lb.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va.sqrt() * vb.sqrt()).max(1e-12)
}

/// Lag (in 10 ms frames) at which `mic` best follows `reference`, searched up to 4 s.
fn best_lag(reference: &[f64], mic: &[f64]) -> usize {
    (0..400.min(mic.len().saturating_sub(100)))
        .max_by(|&a, &b| {
            let score = |lag: usize| {
                let n = reference.len().min(mic.len() - lag);
                correlation(&mic[lag..lag + n], &reference[..n])
            };
            score(a).total_cmp(&score(b))
        })
        .unwrap_or(0)
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

/// Prints echo metrics for (raw, aec, loopback) recordings.
fn analyse(takes: &[Take], render_started_ms: Option<f64>) -> anyhow::Result<()> {
    let [raw, aec, lp] = [&takes[0], &takes[1], &takes[2]];

    // Align the mic to the speaker signal by loudness-envelope correlation (robust to room
    // reverb and Bluetooth codecs, unlike a sample-level fit).
    let (env_raw, env_aec, env_lp) = (envelope(raw), envelope(aec), envelope(lp));
    let lag = best_lag(&env_lp, &env_raw);
    let n = env_lp
        .len()
        .min(env_raw.len().saturating_sub(lag))
        .min(env_aec.len().saturating_sub(lag));
    anyhow::ensure!(n > 200, "recordings too short to analyse");
    let lp = &env_lp[..n];
    let mut sorted = lp.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (loud_t, pause_t) = (sorted[n * 70 / 100], sorted[n * 15 / 100]);
    let stats = |env: &[f64]| {
        let m = &env[lag..lag + n];
        // Medians, not means: room-noise bursts (a cough, a door) would dominate a power mean.
        let median = |pred: &dyn Fn(f64) -> bool| {
            let mut v: Vec<f64> = m
                .iter()
                .zip(lp)
                .filter(|(_, l)| pred(**l))
                .map(|(e, _)| *e)
                .collect();
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(1e-12)
        };
        let (loud, pause) = (median(&|l| l >= loud_t), median(&|l| l <= pause_t));
        // Echo power = level while the clip is loud minus the floor in its pauses.
        let echo = loud - pause;
        (to_db(loud), to_db(pause), echo, correlation(m, lp))
    };
    let (raw_loud, raw_pause, raw_echo, raw_corr) = stats(&env_raw);
    let (aec_loud, aec_pause, aec_echo, aec_corr) = stats(&env_aec);
    // The mic stream started at ~0 and the render stream at `render_started`; the remainder is
    // the acoustic path, including any Bluetooth buffering.
    let delay_ms = render_started_ms.map(|r| lag as f64 * 10.0 - r);

    println!(
        "
{:<5} {:>11} {:>12} {:>14}",
        "", "clip loud", "clip pauses", "follows clip"
    );
    println!(
        "{:<5} {:>8.1} dB {:>9.1} dB {:>14.2}",
        "raw", raw_loud, raw_pause, raw_corr
    );
    println!(
        "{:<5} {:>8.1} dB {:>9.1} dB {:>14.2}",
        "aec", aec_loud, aec_pause, aec_corr
    );
    match delay_ms {
        Some(d) => println!("\nspeaker → mic delay (incl. Bluetooth): ~{d:.0} ms"),
        None => println!("\nspeaker → mic delay: unknown (no meta.json)"),
    }
    println!("alignment: mic lags the speaker signal by {} ms", lag * 10);
    if raw_echo <= 0.0 {
        println!(
            "echo reduction:        not measurable (no echo above the room floor in the raw mic)"
        );
    } else if aec_echo <= 0.0 {
        println!(
            "echo reduction:        > {:.1} dB (residual below the AEC noise floor)",
            to_db(raw_echo) - aec_pause
        );
    } else {
        println!(
            "echo reduction:        {:.1} dB   (good AEC: 25+ dB)",
            to_db(raw_echo) - to_db(aec_echo)
        );
    }
    println!("noise floor reduction: {:.1} dB", raw_pause - aec_pause);
    if raw_echo <= 0.0 || to_db(raw_echo) - raw_pause < 6.0 {
        println!("
WARNING: the clip was barely louder than the room in the raw mic; turn the speakers up or reduce room noise.");
    }
    Ok(())
}

fn load_take(dir: &Path, name: &'static str) -> anyhow::Result<Take> {
    let mut r = hound::WavReader::open(dir.join(format!("{name}.wav")))?;
    let rate = r.spec().sample_rate;
    let samples = r.samples::<f32>().collect::<Result<_, _>>()?;
    Ok(Take {
        name,
        rate,
        samples,
    })
}

fn main() -> anyhow::Result<()> {
    if let Some(dir) = arg("--analyse") {
        let dir = PathBuf::from(dir);
        let takes = vec![
            load_take(&dir, "raw")?,
            load_take(&dir, "aec")?,
            load_take(&dir, "loopback")?,
        ];
        let meta: Option<serde_json::Value> = std::fs::read_to_string(dir.join("meta.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());
        return analyse(&takes, meta.and_then(|m| m["render_started_ms"].as_f64()));
    }

    wasapi::com_init()?;
    // Devices come from config/audio.json unless overridden on the command line.
    let prefs = wasapi::DevicePrefs::load(Path::new("config/audio.json"))?;
    let mic = match arg("--mic") {
        Some(n) => wasapi::find(Flow::Capture, &[&n])?,
        None => prefs.microphone()?,
    };
    let speakers = match arg("--speakers") {
        Some(n) => wasapi::find(Flow::Render, &[&n])?,
        None => prefs.speakers()?,
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

    // WASAPI loopback delivers no packets while nothing renders, so render digital silence
    // around the clip: the loopback recording then covers the whole render timeline.
    let mut render = Render::open(&speakers)?;
    let rate = render.rate() as usize;
    let pad = |d: Duration| rate * d.as_millis() as usize / 1000;
    let mut pcm = vec![0.0; pad(BASELINE)];
    pcm.extend(wasapi::resample_linear(&clip, clip_rate, render.rate()));
    pcm.extend(std::iter::repeat_n(0.0, pad(TAIL)));
    let render_started = started.elapsed();
    println!("playing {:.1} s ...", pcm.len() as f64 / rate as f64);
    render.play_blocking(&pcm)?;
    thread::sleep(Duration::from_millis(500)); // let the Bluetooth/room tail arrive
    stop.store(true, Ordering::Relaxed);

    let takes: Vec<Take> = takes
        .into_iter()
        .map(|h| h.join().expect("capture thread panicked"))
        .collect::<anyhow::Result<_>>()?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let dir = PathBuf::from(format!("bench-results/aec/{stamp}"));
    std::fs::create_dir_all(&dir)?;
    for take in &takes {
        save(&dir, take)?;
    }
    std::fs::write(
        dir.join("meta.json"),
        format!("{{\"render_started_ms\": {}}}", render_started.as_millis()),
    )?;
    analyse(&takes, Some(render_started.as_secs_f64() * 1000.0))?;
    println!("recordings: {}", dir.display());
    Ok(())
}
