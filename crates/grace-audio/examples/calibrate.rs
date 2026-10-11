//! Speaker calibration (issue #33): measures speaker→mic delay, its stability and echo level.
//!
//!   cargo run --release -p grace-audio --example calibrate -- --play
//!
//! Plays three short sweeps (~6 s total) on the speakers from `config/audio.json` while
//! recording the raw mic. Keep the room quiet. Without `--play` it only shows the devices;
//! `--check` opens the streams without playing anything.
//! Results go to `data/calibration.json` (one entry per speaker); the recordings go to
//! `bench-results/calibration/<time>/` for offline echo-canceller work.

use std::{collections::BTreeMap, path::Path};

use grace_audio::{
    calibrate::{self, Calibration},
    wasapi::{self, DevicePrefs},
};

fn save_wav(path: &Path, rate: u32, samples: &[f32]) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    wasapi::com_init()?;
    let prefs = DevicePrefs::load(Path::new("config/audio.json"))?;
    let (mic, speakers) = (prefs.microphone()?, prefs.speakers()?);
    println!("mic:      {}\nspeakers: {}", mic.name, speakers.name);
    if std::env::args().any(|a| a == "--check") {
        // Opens every stream calibration needs without starting playback: silent.
        use grace_audio::wasapi::{Capture, CaptureMode, Render};
        Render::open(&speakers)?;
        Capture::open(&mic, &CaptureMode::Raw)?;
        Capture::open(&speakers, &CaptureMode::Loopback)?;
        println!(
            "
check: all streams open; calibration can run."
        );
        return Ok(());
    }
    if !std::env::args().any(|a| a == "--play") {
        println!("\nDry run. Add --play to play the calibration sweeps (~6 s).");
        return Ok(());
    }
    println!("calibrating: keep the room quiet ...");
    let (cal, rec) = calibrate::run(&mic, &speakers)?;

    println!("\ndelay (playback → mic):  {:.0} ms", cal.delay_ms);
    let per: Vec<String> = cal.delays_ms.iter().map(|d| format!("{d:.0}")).collect();
    println!(
        "  per sweep:             {} ms (spread {:.1} ms)",
        per.join(", "),
        cal.delay_spread_ms
    );
    println!("room noise:              {:.1} dBFS", cal.noise_floor_dbfs);
    println!(
        "echo while playing:      {:.1} dBFS ({:+.1} dB over the room)",
        cal.echo_dbfs,
        cal.echo_dbfs - cal.noise_floor_dbfs
    );
    println!("echo detection SNR:      {:.1} dB", cal.echo_snr_db);
    for note in cal.verdict() {
        println!("note: {note}");
    }

    let dir = Path::new("bench-results/calibration").join(cal.measured_at_unix.to_string());
    std::fs::create_dir_all(&dir)?;
    save_wav(&dir.join("mic.wav"), rec.rate, &rec.mic)?;
    save_wav(&dir.join("loopback.wav"), rec.rate, &rec.loopback)?;
    std::fs::write(
        dir.join("timelines.json"),
        serde_json::to_string(
            &serde_json::json!({ "mic": rec.mic_timeline, "loopback": rec.loopback_timeline }),
        )?,
    )?;
    std::fs::write(
        dir.join("calibration.json"),
        serde_json::to_string_pretty(&cal)?,
    )?;

    let store = Path::new("data/calibration.json");
    std::fs::create_dir_all("data")?;
    let mut all: BTreeMap<String, Calibration> = std::fs::read_to_string(store)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    all.insert(cal.speakers.clone(), cal);
    std::fs::write(store, serde_json::to_string_pretty(&all)?)?;
    println!("\nsaved {} and {}", store.display(), dir.display());
    Ok(())
}
