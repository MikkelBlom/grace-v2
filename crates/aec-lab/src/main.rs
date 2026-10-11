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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        args.get(1)
            .context("usage: aec-lab <dir> [offset ms] [lead ms]")?,
    );
    let calibrated = dir.join("timelines.json").exists();
    let mic_file = if calibrated { "mic.wav" } else { "raw.wav" };
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
        let delay = cal["delay_ms"].as_f64().context("delay_ms")?;
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

    // render[k] carries what the mic will hear `lead` later: render[k] = lp[k + offset - lead].
    let shift = ((offset_ms - lead_ms) / 1000.0 * rate as f64).round() as i64;
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
    let mut pipeline = linear::builder(format, format)
        .initial_delay_ms(lead_ms as i32)
        .export_metrics(true)
        .build()?;
    let frame = format.sample_count();
    let mut out = Vec::with_capacity(mic.len());
    let mut buf = vec![0.0f32; frame];
    for (cap, ren) in mic.chunks_exact(frame).zip(render.chunks_exact(frame)) {
        pipeline.handle_render_frame(ren)?;
        if pipeline.process_capture_frame(cap, &mut buf)? {
            out.extend_from_slice(&buf);
        }
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
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
