//! Offline echo-cancellation lab: replays a recorded `aec_test` run (raw mic + speaker loopback)
//! through our own WebRTC AEC3 pipeline, so AEC settings can be tuned without playing sound.
//!
//!   cargo run --release -p aec-lab -- <run dir> [offset ms] [lead ms]
//!
//! A calibration run (`mic.wav`, `loopback.wav`, `timelines.json`, `calibration.json`) is aligned
//! exactly: both streams are placed on the shared QPC clock and shifted by the measured delay.
//! An older `aec_test` run (`raw.wav`, `loopback.wav`) needs `offset ms`: where loopback content
//! shows up in the mic recording (mic[t] = loopback[t + offset]), e.g. from GCC-PHAT.
//! `lead ms` is how far the reference is fed ahead of the capture, leaving AEC3's delay
//! estimator a small positive delay to find. Writes `ours.wav` next to the inputs.

use std::path::{Path, PathBuf};

use aec3::{nodes::audio::AudioFormat, pipelines::linear};
use anyhow::Context;
use grace_audio::wasapi::Timeline;

fn read(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| path.display().to_string())?;
    let rate = r.spec().sample_rate;
    Ok((r.samples::<f32>().collect::<Result<_, _>>()?, rate))
}

/// Echo reduction on the frames where the mic hears echo (well above its quietest tenth), per
/// third of the recording so convergence is visible.
fn report(mic: &[f32], out: &[f32], rate: u32) {
    let f = rate as usize / 100;
    let n = mic.len().min(out.len()) / f;
    let energy = |x: &[f32], i: usize| {
        x[i * f..(i + 1) * f]
            .iter()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
    };
    let mic_e: Vec<f64> = (0..n).map(|i| energy(mic, i)).collect();
    let mut sorted = mic_e.clone();
    sorted.sort_by(f64::total_cmp);
    let floor = sorted[n / 10].max(1e-12);
    for part in 0..3 {
        let (from, to) = (part * n / 3, (part + 1) * n / 3);
        let echo: Vec<usize> = (from..to).filter(|&i| mic_e[i] > floor * 30.0).collect();
        if echo.is_empty() {
            println!("third {}: no echo frames", part + 1);
            continue;
        }
        let m: f64 = echo.iter().map(|&i| mic_e[i]).sum();
        let o: f64 = echo.iter().map(|&i| energy(out, i)).sum();
        println!(
            "third {}: {} echo frames, reduction {:.1} dB",
            part + 1,
            echo.len(),
            10.0 * (m / o.max(1e-12)).log10()
        );
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        args.get(1)
            .context("usage: aec-lab <dir> [offset ms] [lead ms]")?,
    );
    let calibrated = dir.join("timelines.json").exists();
    let calibrated = calibrated && dir.join("calibration.json").exists();
    // Calibration runs record `mic.wav`; aec_test runs record `raw.wav` (plus Windows' `aec.wav`).
    let mic_file = if dir.join("mic.wav").exists() {
        "mic.wav"
    } else {
        "raw.wav"
    };
    let (mic, rate) = read(&dir.join(mic_file))?;
    let (lp, lp_rate) = read(&dir.join("loopback.wav"))?;
    anyhow::ensure!(rate == lp_rate, "mic {rate} Hz vs loopback {lp_rate} Hz");
    let (offset_ms, lead_ms) = if calibrated {
        let tl: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("timelines.json"))?)?;
        let cal: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("calibration.json"))?)?;
        let mic_tl: Timeline = serde_json::from_value(tl["mic"].clone())?;
        let lp_tl: Timeline = serde_json::from_value(tl["loopback"].clone())?;
        // mic sample k was captured at mic_t0 + k/rate; the loopback sample captured at that same
        // instant is index k + (mic_t0 - lp_t0)*rate; the echo it causes arrives `delay` later.
        let clock_ms = (mic_tl.time_of(0.0).context("mic timeline")?
            - lp_tl.time_of(0.0).context("loopback timeline")?)
            * 1000.0;
        // `--delay <ms>` overrides the calibration (Bluetooth delay changes between sessions).
        let delay = match args.iter().position(|a| a == "--delay") {
            Some(i) => args.get(i + 1).context("--delay <ms>")?.parse()?,
            None => cal["delay_ms"].as_f64().context("delay_ms")?,
        };
        let lead = args.get(2).map_or(Ok(30.0), |s| s.parse())?;
        println!("calibrated: clock offset {clock_ms:.1} ms, echo delay {delay:.1} ms");
        (clock_ms - delay, lead)
    } else {
        let offset: f64 = args
            .get(2)
            .context("offset ms (uncalibrated run)")?
            .parse()?;
        (offset, args.get(3).map_or(Ok(60.0), |s| s.parse())?)
    };

    // mic[k] holds the echo of lp[k + offset]. AEC3 needs the reference *before* the echo, so
    // render[k] = what the mic will hear `lead` later = lp[k + lead + offset].
    let shift = ((offset_ms + lead_ms) / 1000.0 * rate as f64).round() as i64;
    let render: Vec<f32> = (0..mic.len() as i64)
        .map(|k| {
            let j = k + shift;
            if j >= 0 && (j as usize) < lp.len() {
                lp[j as usize]
            } else {
                0.0
            }
        })
        .collect();

    let format = AudioFormat::ten_ms(rate, 1);
    let flag = |f: &str| args.iter().any(|a| a == f);
    let mut pipeline = linear::builder(format, format)
        .initial_delay_ms(lead_ms as i32)
        .enable_gain_controller2(!flag("--no-agc"))
        .enable_noise_suppression(!flag("--no-ns"))
        .export_metrics(true)
        .export_linear_output(true)
        .build()?;
    let frame = format.sample_count();
    // `--warm` runs the recording through once first and keeps only the second pass: the state
    // a continuously running Grace would be in, with the canceller already adapted to the room.
    let passes = if flag("--warm") { 2 } else { 1 };
    let mut out = Vec::with_capacity(mic.len());
    let (mut linear, mut linear_frame) = (Vec::new(), 0usize);
    let mut buf = vec![0.0f32; frame];
    for pass in 0..passes {
        out.clear();
        linear.clear();
        for (cap, ren) in mic.chunks_exact(frame).zip(render.chunks_exact(frame)) {
            pipeline.handle_render_frame(ren)?;
            if pipeline.process_capture_frame(cap, &mut buf)? {
                out.extend_from_slice(&buf);
            }
            // The linear filter's output, before residual suppression: keeps near-end speech intact.
            while let Some(chunk) = pipeline.try_pull_linear_output()? {
                linear_frame = chunk.payload().samples().len();
                linear.extend_from_slice(chunk.payload().samples());
            }
        }
        let _ = pass;
    }
    let mut metrics = None;
    while let Some(m) = pipeline.try_pull_metrics()? {
        metrics = Some(*m.payload());
    }
    if let Some(m) = metrics {
        println!(
            "aec3 metrics: ERL {:.1} dB, ERLE {:.1} dB, estimated delay {} ms",
            m.echo_return_loss, m.echo_return_loss_enhancement, m.delay_ms
        );
    }
    report(&mic, &out, rate);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    if !linear.is_empty() {
        let linear_rate = (linear_frame * 100) as u32;
        let mut w = hound::WavWriter::create(
            dir.join("ours_linear.wav"),
            hound::WavSpec { sample_rate: linear_rate, ..spec },
        )?;
        for s in &linear {
            w.write_sample(*s)?;
        }
        w.finalize()?;
        println!("wrote ours_linear.wav ({linear_rate} Hz)");
    }
    let path = dir.join("ours.wav");
    let mut w = hound::WavWriter::create(&path, spec)?;
    for s in &out {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    println!(
        "wrote {} ({:.1} s)",
        path.display(),
        out.len() as f64 / rate as f64
    );
    Ok(())
}
