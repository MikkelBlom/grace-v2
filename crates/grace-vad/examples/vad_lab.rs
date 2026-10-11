//! Runs Silero VAD + the gate over WAV files and prints one JSON line per file:
//! speech segments (seconds), total speech, and CPU cost per 32 ms frame.
//!
//!   cargo run --release -p grace-vad --example vad_lab -- [--model <onnx>] [--grace-talking] [--boost <x>] <file.wav>...
//!
//! `--grace-talking` treats the whole file as "Grace is speaking" (raised start threshold).

use std::{path::PathBuf, time::Instant};

use anyhow::Context;
use grace_core::vad_gate::{VadEdge, VadGate, VadGateConfig};
use grace_vad::{Decimate3, FRAME, SileroVad};
use serde_json::json;

fn load_16k(path: &PathBuf) -> anyhow::Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| path.display().to_string())?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = raw
        .chunks_exact(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    match spec.sample_rate {
        16_000 => Ok(mono),
        48_000 => {
            let mut out = Vec::with_capacity(mono.len() / 3);
            Decimate3::default().push(&mono, &mut out);
            Ok(out)
        }
        other => anyhow::bail!(
            "{}: {other} Hz not supported (16 or 48 kHz)",
            path.display()
        ),
    }
}

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let model = match args.iter().position(|a| a == "--model") {
        Some(i) => {
            args.remove(i);
            PathBuf::from(args.remove(i))
        }
        None => PathBuf::from(".runtime/models/silero_vad_v6.2.3.onnx"),
    };
    let take_flag = |args: &mut Vec<String>, f: &str| match args.iter().position(|a| a == f) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    };
    let grace_talking = take_flag(&mut args, "--grace-talking");
    let mut cfg = VadGateConfig::default();
    if let Some(i) = args.iter().position(|a| a == "--boost") {
        args.remove(i);
        cfg.talking_boost = args.remove(i).parse()?;
    }
    let mut vad = SileroVad::new(&model)?;
    for file in args.iter().map(PathBuf::from) {
        let audio = load_16k(&file)?;
        vad.reset();
        let mut gate = VadGate::new(cfg.clone());
        gate.set_grace_talking(grace_talking);
        let secs = |frame: u64| frame as f64 * FRAME as f64 / 16_000.0;
        let (mut segments, mut open) = (Vec::new(), None);
        let started = Instant::now();
        let frames = audio.len() / FRAME;
        for chunk in audio.chunks_exact(FRAME) {
            match gate.push(vad.process(chunk)?) {
                Some(VadEdge::Started { frame }) => open = Some(secs(frame)),
                Some(VadEdge::Stopped { frame }) => {
                    if let Some(s) = open.take() {
                        segments.push([s, secs(frame)]);
                    }
                }
                None => {}
            }
        }
        if let Some(s) = open {
            segments.push([s, secs(frames as u64)]);
        }
        let speech: f64 = segments.iter().map(|[a, b]| b - a).sum();
        println!(
            "{}",
            json!({
                "file": file,
                "duration_s": secs(frames as u64),
                "speech_s": speech,
                "segments": segments,
                "us_per_frame": started.elapsed().as_secs_f64() * 1e6 / frames.max(1) as f64,
            })
        );
    }
    Ok(())
}
