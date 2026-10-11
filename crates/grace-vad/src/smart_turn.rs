//! End-of-turn detection: Pipecat Smart Turn v3.2 (audio-only, 23 languages incl. Danish).
//!
//! Given the user's speech so far, it returns the probability that they have finished their
//! turn, judged from prosody rather than silence length. It looks at the last 8 s; shorter audio
//! is zero-padded at the start, exactly like the reference implementation.

use std::path::Path;

use anyhow::Context;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

use crate::mel::{LogMel, N_MELS, normalize};

const WINDOW: usize = 8 * 16_000;

pub struct SmartTurn {
    session: Session,
    mel: LogMel,
}

impl SmartTurn {
    pub fn new(model: &Path) -> anyhow::Result<Self> {
        let flat = |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow::anyhow!("{e}");
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(flat)?
            .with_intra_threads(2)
            .map_err(flat)?
            .commit_from_file(model)
            .with_context(|| format!("loading {}", model.display()))?;
        Ok(Self {
            session,
            mel: LogMel::default(),
        })
    }

    /// Features exactly as the model was trained on, for the last 8 s of 16 kHz `audio`.
    pub fn features(&self, audio: &[f32]) -> Vec<f32> {
        let tail = &audio[audio.len().saturating_sub(WINDOW)..];
        let mut window = vec![0.0f32; WINDOW - tail.len()];
        window.extend_from_slice(tail);
        normalize(&mut window);
        self.mel.compute(&window)
    }

    /// Probability (0–1) that the speaker has finished their turn.
    pub fn predict(&mut self, audio: &[f32]) -> anyhow::Result<f32> {
        let features = self.features(audio);
        let frames = features.len() / N_MELS;
        let outputs = self
            .session
            .run(ort::inputs!["input_features" => Tensor::from_array(([1usize, N_MELS, frames], features))?])?;
        let (_, out) = outputs["logits"].try_extract_tensor::<f32>()?;
        Ok(out[0])
    }
}
