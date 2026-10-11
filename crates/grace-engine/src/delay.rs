//! Echo-delay tracking: GCC-PHAT between what the speakers played (loopback) and what the mic
//! heard. Bluetooth delay differs per session and drifts (110–165 ms measured), so the engine
//! re-estimates it continuously from whatever is playing.

use rustfft::{FftPlanner, num_complex::Complex};

/// Lag (samples) by which `mic` trails `reference`, searched in `0..=max_lag`, plus how far the
/// correlation peak stands out (in standard deviations; > ~6 is trustworthy).
pub fn gcc_phat(reference: &[f32], mic: &[f32], max_lag: usize) -> (usize, f32) {
    let n = (reference.len() + mic.len()).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let (fwd, inv) = (planner.plan_fft_forward(n), planner.plan_fft_inverse(n));
    let to_c = |x: &[f32]| {
        let mut v: Vec<Complex<f32>> = x.iter().map(|&r| Complex::new(r, 0.0)).collect();
        v.resize(n, Complex::new(0.0, 0.0));
        v
    };
    let (mut a, mut b) = (to_c(mic), to_c(reference));
    fwd.process(&mut a);
    fwd.process(&mut b);
    for (x, y) in a.iter_mut().zip(&b) {
        let c = *x * y.conj();
        *x = c / (c.norm() + 1e-12);
    }
    inv.process(&mut a);
    let cc: Vec<f32> = a[..=max_lag.min(n - 1)]
        .iter()
        .map(|c| c.re.abs())
        .collect();
    let (lag, peak) = cc
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .map(|(i, v)| (i, *v))
        .unwrap_or((0, 0.0));
    let mean = cc.iter().sum::<f32>() / cc.len() as f32;
    let sd = (cc.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cc.len() as f32).sqrt();
    (lag, (peak - mean) / sd.max(1e-12))
}

/// Median of recent trusted estimates, so one bad window can't move the alignment.
#[derive(Debug, Clone)]
pub struct DelayTracker {
    recent: Vec<f64>,
    pub current_ms: f64,
}

impl DelayTracker {
    pub fn new(initial_ms: f64) -> Self {
        Self {
            recent: Vec::new(),
            current_ms: initial_ms,
        }
    }

    /// Feeds one estimate; returns the new delay if it moved by more than 2 ms.
    pub fn update(&mut self, ms: f64, confidence: f32) -> Option<f64> {
        if confidence < 6.0 {
            return None;
        }
        self.recent.push(ms);
        if self.recent.len() > 5 {
            self.recent.remove(0);
        }
        let mut sorted = self.recent.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        if (median - self.current_ms).abs() > 2.0 {
            self.current_ms = median;
            Some(median)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_known_delay_in_noise() {
        let mut x = 12345u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 2000) as f32 / 1000.0 - 1.0
        };
        let reference: Vec<f32> = (0..16_000).map(|_| rnd()).collect();
        let delay = 2_480;
        let mic: Vec<f32> = (0..16_000)
            .map(|i| if i >= delay { 0.3 * reference[i - delay] } else { 0.0 } + 0.2 * rnd())
            .collect();
        let (lag, conf) = gcc_phat(&reference, &mic, 8_000);
        assert_eq!(lag, delay);
        assert!(conf > 6.0, "confidence {conf}");
    }

    #[test]
    fn tracker_ignores_outliers_and_low_confidence() {
        let mut t = DelayTracker::new(110.0);
        assert_eq!(t.update(400.0, 3.0), None, "low confidence");
        assert_eq!(t.update(160.0, 9.0), Some(160.0));
        assert_eq!(t.update(161.0, 9.0), None);
        assert_eq!(
            t.update(398.0, 9.0),
            None,
            "single outlier doesn't move the median"
        );
        assert_eq!(t.current_ms, 160.0);
    }
}
