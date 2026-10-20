//! Audio rings indexed by capture time, so streams from different devices can be lined up.
//!
//! Times are seconds on the QPC clock (see `clock`). A gap in incoming timestamps (WASAPI
//! loopback delivers nothing while the speakers are idle) is filled with silence, so reading
//! "what was playing at time t" stays correct.

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct TimedRing {
    rate: f64,
    /// Capture time of `buf[0]`.
    start: f64,
    buf: VecDeque<f32>,
    capacity: usize,
    /// While set, samples from this time on are kept past `capacity`: an open turn is read
    /// back whole however long it runs (live run 4: a 65 s note lost its start at 30 s).
    keep_from: Option<f64>,
}

/// Even a pinned ring never holds more than this (a pin left set by a bug must not eat RAM).
const MAX_PINNED_SECONDS: f64 = 30.0 * 60.0;

impl TimedRing {
    pub fn new(rate: u32, seconds: f64) -> Self {
        Self {
            rate: rate as f64,
            start: 0.0,
            buf: VecDeque::new(),
            capacity: (seconds * rate as f64) as usize,
            keep_from: None,
        }
    }

    /// Keep everything from `t` on (an open turn) until called with `None`.
    pub fn keep_from(&mut self, t: Option<f64>) {
        self.keep_from = t;
    }

    /// Time just past the last sample.
    pub fn end(&self) -> f64 {
        self.start + self.buf.len() as f64 / self.rate
    }

    /// Appends `samples` whose first sample was captured at `t`.
    pub fn push(&mut self, t: f64, samples: &[f32]) {
        if self.buf.is_empty() {
            self.start = t;
        } else {
            // Only real gaps are filled; sub-2 ms timestamp jitter is ignored so a continuous
            // stream stays sample-continuous.
            let gap = t - self.end();
            if gap > 0.002 {
                let n = (gap * self.rate).round() as usize;
                self.buf
                    .extend(std::iter::repeat_n(0.0, n.min(self.capacity)));
            }
        }
        self.buf.extend(samples.iter().copied());
        let mut excess = self.buf.len().saturating_sub(self.capacity);
        if let Some(keep) = self.keep_from {
            let before_keep = ((keep - self.start) * self.rate).max(0.0) as usize;
            let hard = self
                .buf
                .len()
                .saturating_sub((MAX_PINNED_SECONDS * self.rate) as usize);
            excess = excess.min(before_keep).max(hard);
        }
        if excess > 0 {
            self.buf.drain(..excess);
            self.start += excess as f64 / self.rate;
        }
    }

    /// `n` samples starting at time `t`; silence where the ring has no data.
    pub fn read(&self, t: f64, n: usize) -> Vec<f32> {
        let first = ((t - self.start) * self.rate).round() as i64;
        (0..n as i64)
            .map(|i| {
                let j = first + i;
                if j >= 0 && (j as usize) < self.buf.len() {
                    self.buf[j as usize]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Samples between times `t0` and `t1`.
    pub fn read_span(&self, t0: f64, t1: f64) -> Vec<f32> {
        let n = ((t1 - t0).max(0.0) * self.rate) as usize;
        self.read(t0, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_by_time_and_fills_gaps_with_silence() {
        let mut r = TimedRing::new(100, 10.0);
        r.push(1.0, &[1.0; 10]); // 1.00..1.10
        r.push(1.5, &[2.0; 10]); // gap 1.10..1.50, then 1.50..1.60
        assert!((r.end() - 1.6).abs() < 1e-9);
        assert_eq!(r.read(1.05, 3), vec![1.0; 3]);
        assert_eq!(r.read(1.2, 2), vec![0.0; 2]);
        assert_eq!(r.read(1.55, 3), vec![2.0, 2.0, 2.0]);
        assert_eq!(r.read(0.5, 2), vec![0.0; 2], "before the ring");
    }

    #[test]
    fn keeps_an_open_turn_past_capacity() {
        let mut r = TimedRing::new(100, 1.0); // holds 1 s
        r.push(0.0, &[1.0; 50]);
        r.keep_from(Some(0.5)); // a turn starts at 0.5 s
        for i in 1..5 {
            r.push(i as f64 * 0.5, &[2.0; 50]); // talks for 2 more seconds
        }
        assert_eq!(r.read_span(0.5, 2.5), vec![2.0; 200], "whole turn kept");
        assert_eq!(r.read(0.0, 1), vec![0.0], "older audio still evicted");
        r.keep_from(None);
        r.push(2.5, &[3.0; 10]);
        assert!(
            (r.end() - r.start - 1.0).abs() < 1e-9,
            "back to 1 s once the turn ends"
        );
    }

    #[test]
    fn small_jitter_keeps_continuity() {
        let mut r = TimedRing::new(1000, 10.0);
        r.push(0.0, &[1.0; 10]);
        r.push(0.0105, &[2.0; 10]); // 0.5 ms late timestamp: no gap inserted
        assert_eq!(r.read(0.0, 20).iter().filter(|&&v| v == 0.0).count(), 0);
    }

    #[test]
    fn capacity_drops_oldest_and_advances_start() {
        let mut r = TimedRing::new(10, 1.0);
        r.push(0.0, &(0..15).map(|i| i as f32).collect::<Vec<_>>());
        assert!((r.end() - 1.5).abs() < 1e-9);
        assert_eq!(r.read(0.5, 1), vec![5.0]);
    }
}
