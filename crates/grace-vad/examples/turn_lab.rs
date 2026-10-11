//! Smart Turn on WAV files (16 kHz mono): prints the end-of-turn probability, and with
//! `--dump <file>` writes the 80×800 feature matrix (f32 LE) for checking against Python.
//!
//!   cargo run --release -p grace-vad --example turn_lab -- [--dump feats.bin] <file.wav>...

use std::{path::PathBuf, time::Instant};

use grace_vad::smart_turn::SmartTurn;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let dump = args.iter().position(|a| a == "--dump").map(|i| {
        args.remove(i);
        PathBuf::from(args.remove(i))
    });
    let mut st = SmartTurn::new(&PathBuf::from(".runtime/models/smart-turn-v3.2-fp32.onnx"))?;
    for file in &args {
        let mut r = hound::WavReader::open(file)?;
        anyhow::ensure!(
            r.spec().sample_rate == 16_000 && r.spec().channels == 1,
            "{file}: need 16 kHz mono"
        );
        let audio: Vec<f32> = match r.spec().sample_format {
            hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => r
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?,
        };
        if let Some(path) = &dump {
            let bytes: Vec<u8> = st
                .features(&audio)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(path, bytes)?;
        }
        let started = Instant::now();
        let p = st.predict(&audio)?;
        println!(
            "{file}\t{p:.4}\t{:.1} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}
