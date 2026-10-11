//! Whisper-style log-mel features, matching Hugging Face's `WhisperFeatureExtractor`
//! (16 kHz, 400-sample Hann window, hop 160, 80 Slaney mel bins up to 8 kHz, log10, clamped to
//! 8 dB below the max, scaled to roughly [-1, 1]). Verified against the Python implementation
//! (see the test and `bench/mel_check.py`).

use rustfft::{FftPlanner, num_complex::Complex};

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 80;
const RATE: f64 = 16_000.0;

fn hz_to_mel(f: f64) -> f64 {
    // Slaney scale: linear below 1 kHz, logarithmic above.
    if f < 1000.0 {
        3.0 * f / 200.0
    } else {
        15.0 + (f / 1000.0).ln() * 27.0 / 6.4f64.ln()
    }
}

fn mel_to_hz(m: f64) -> f64 {
    if m < 15.0 {
        200.0 * m / 3.0
    } else {
        1000.0 * ((m - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

/// `[N_MELS][N_FFT/2 + 1]` triangular filters with Slaney area normalization.
fn mel_filters() -> Vec<Vec<f32>> {
    let bins = N_FFT / 2 + 1;
    let fft_freqs: Vec<f64> = (0..bins).map(|k| k as f64 * RATE / N_FFT as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
    let pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    (0..N_MELS)
        .map(|m| {
            let (f0, f1, f2) = (pts[m], pts[m + 1], pts[m + 2]);
            let norm = 2.0 / (f2 - f0);
            fft_freqs
                .iter()
                .map(|&f| {
                    let up = (f - f0) / (f1 - f0);
                    let down = (f2 - f) / (f2 - f1);
                    (up.min(down).max(0.0) * norm) as f32
                })
                .collect()
        })
        .collect()
}

pub struct LogMel {
    filters: Vec<Vec<f32>>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Default for LogMel {
    fn default() -> Self {
        // Periodic Hann, as numpy/torch use for STFT.
        let window = (0..N_FFT)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32
            })
            .collect();
        Self {
            filters: mel_filters(),
            window,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }
}

impl LogMel {
    /// Features for 16 kHz `audio`, as `N_MELS` rows of `audio.len() / HOP` frames, row-major.
    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        // center=True with reflect padding by N_FFT/2 on both sides.
        let pad = N_FFT / 2;
        let n = audio.len();
        let reflect = |i: isize| -> f32 {
            let mut j = i;
            if j < 0 {
                j = -j;
            }
            if j >= n as isize {
                j = 2 * (n as isize - 1) - j;
            }
            audio[j.clamp(0, n as isize - 1) as usize]
        };
        let padded: Vec<f32> = (-(pad as isize)..(n + pad) as isize).map(reflect).collect();
        // Whisper drops the last STFT frame.
        let frames = 1 + (padded.len() - N_FFT) / HOP - 1;
        let bins = N_FFT / 2 + 1;
        let mut mel = vec![0.0f32; N_MELS * frames];
        let mut buf = vec![Complex::new(0.0f32, 0.0); N_FFT];
        let mut power = vec![0.0f32; bins];
        for t in 0..frames {
            let start = t * HOP;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (k, p) in power.iter_mut().enumerate() {
                *p = buf[k].norm_sqr();
            }
            for (m, filter) in self.filters.iter().enumerate() {
                let e: f32 = filter.iter().zip(&power).map(|(a, b)| a * b).sum();
                mel[m * frames + t] = e.max(1e-10).log10();
            }
        }
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in mel.iter_mut() {
            *v = ((*v).max(max - 8.0) + 4.0) / 4.0;
        }
        mel
    }
}

/// Zero-mean, unit-variance normalization over the whole buffer (Hugging Face `do_normalize`).
pub fn normalize(audio: &mut [f32]) {
    let n = audio.len().max(1) as f64;
    let mean = audio.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = audio
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let scale = 1.0 / (var + 1e-7).sqrt();
    for v in audio.iter_mut() {
        *v = ((*v as f64 - mean) * scale) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_seconds_give_800_frames_of_80_bins() {
        let audio: Vec<f32> = (0..128_000)
            .map(|i| (i as f32 * 0.05).sin() * 0.1)
            .collect();
        let f = LogMel::default().compute(&audio);
        assert_eq!(f.len(), N_MELS * 800);
        assert!(f.iter().all(|v| v.is_finite() && *v <= 1.5 && *v >= -1.5));
    }

    #[test]
    fn mel_scale_round_trips() {
        for f in [0.0, 500.0, 1000.0, 3000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6);
        }
    }
}
