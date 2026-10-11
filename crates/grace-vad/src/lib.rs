//! Speech-model wrappers on ONNX Runtime: Silero VAD v6 (is someone speaking?), Smart Turn
//! v3.2 (have they finished?), plus a 48→16 kHz decimator and Whisper log-mel features.
//!
//! [`SileroVad`] returns a speech probability per 32 ms frame. Turning probabilities into
//! "speech started / stopped" edges is policy, and lives in `grace_core::vad_gate`.

pub mod mel;
pub mod smart_turn;

use std::path::Path;

use anyhow::Context;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

/// Samples per VAD frame at 16 kHz (32 ms).
pub const FRAME: usize = 512;
const CONTEXT: usize = 64;
const RATE: i64 = 16_000;

pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    context: [f32; CONTEXT],
}

impl SileroVad {
    pub fn new(model: &Path) -> anyhow::Result<Self> {
        // Builder errors carry the (non-Send) builder; flatten them to text for anyhow.
        let flat = |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow::anyhow!("{e}");
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(flat)?
            // One thread: a 32 ms frame takes well under a millisecond, and Grace must stay light.
            .with_intra_threads(1)
            .map_err(flat)?
            .commit_from_file(model)
            .with_context(|| format!("loading {}", model.display()))?;
        Ok(Self {
            session,
            state: vec![0.0; 2 * 128],
            context: [0.0; CONTEXT],
        })
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context = [0.0; CONTEXT];
    }

    /// Speech probability for one 512-sample frame of 16 kHz mono audio.
    pub fn process(&mut self, frame: &[f32]) -> anyhow::Result<f32> {
        anyhow::ensure!(
            frame.len() == FRAME,
            "VAD frame must be {FRAME} samples, got {}",
            frame.len()
        );
        // The model expects the previous frame's last 64 samples prepended (as Silero's own wrapper does).
        let mut input = Vec::with_capacity(CONTEXT + FRAME);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(frame);
        self.context.copy_from_slice(&frame[FRAME - CONTEXT..]);
        let outputs = self.session.run(ort::inputs![
            "input" => Tensor::from_array(([1usize, CONTEXT + FRAME], input))?,
            "state" => Tensor::from_array(([2usize, 1, 128], self.state.clone()))?,
            "sr" => Tensor::from_array(((), vec![RATE]))?,
        ])?;
        let (_, prob) = outputs["output"].try_extract_tensor::<f32>()?;
        let (_, state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.state.copy_from_slice(state);
        Ok(prob[0])
    }
}

/// Streaming 3:1 decimator (48 → 16 kHz) with a windowed-sinc low-pass at ~7 kHz.
pub struct Decimate3 {
    taps: Vec<f32>,
    history: Vec<f32>,
    phase: usize,
}

impl Default for Decimate3 {
    fn default() -> Self {
        const N: usize = 47;
        let cutoff = 7_000.0 / 48_000.0;
        let mid = (N / 2) as f64;
        let mut taps: Vec<f32> = (0..N)
            .map(|i| {
                let x = i as f64 - mid;
                let sinc = if x == 0.0 {
                    2.0 * cutoff
                } else {
                    (2.0 * std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                let window =
                    0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (N - 1) as f64).cos();
                (sinc * window) as f32
            })
            .collect();
        let sum: f32 = taps.iter().sum();
        taps.iter_mut().for_each(|t| *t /= sum);
        Self {
            history: vec![0.0; N - 1],
            taps,
            phase: 0,
        }
    }
}

impl Decimate3 {
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let n = self.taps.len();
        for &x in input {
            self.history.push(x);
            if self.phase == 0 {
                let window = &self.history[self.history.len() - n..];
                out.push(
                    window
                        .iter()
                        .zip(self.taps.iter().rev())
                        .map(|(a, b)| a * b)
                        .sum(),
                );
            }
            self.phase = (self.phase + 1) % 3;
            if self.history.len() > 4 * n {
                self.history.drain(..self.history.len() - (n - 1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimator_keeps_low_tones_and_removes_high_ones() {
        let tone = |hz: f64| -> Vec<f32> {
            (0..48_000)
                .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / 48_000.0).sin() as f32)
                .collect()
        };
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        for (hz, keep) in [(1_000.0, true), (12_000.0, false)] {
            let mut d = Decimate3::default();
            let mut out = Vec::new();
            d.push(&tone(hz), &mut out);
            assert_eq!(out.len(), 16_000);
            let level = rms(&out[1000..]);
            if keep {
                assert!((level - 0.707).abs() < 0.02, "{hz} Hz level {level}");
            } else {
                assert!(level < 0.03, "{hz} Hz leaked: {level}");
            }
        }
    }

    #[test]
    fn decimator_is_streaming_consistent() {
        let x: Vec<f32> = (0..9_000)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let (mut a, mut b) = (Decimate3::default(), Decimate3::default());
        let (mut whole, mut chunked) = (Vec::new(), Vec::new());
        a.push(&x, &mut whole);
        for c in x.chunks(481) {
            b.push(c, &mut chunked);
        }
        assert_eq!(whole, chunked);
    }
}
