//! turn_lab with a model argument: turn_any <model.onnx> <file.wav>...
use std::path::PathBuf;
use grace_vad::smart_turn::SmartTurn;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut st = SmartTurn::new(&PathBuf::from(&args[0]))?;
    for file in &args[1..] {
        let mut r = hound::WavReader::open(file)?;
        let audio: Vec<f32> = match r.spec().sample_format {
            hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => r.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?,
        };
        println!("{file}\t{:.4}", st.predict(&audio)?);
    }
    Ok(())
}
