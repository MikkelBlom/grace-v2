//! Speaker calibration: measures the echo path from the speakers back into the microphone.
//!
//! Plays a few exponential sine sweeps, records the raw microphone and the speaker loopback with
//! clock timestamps, and finds each sweep in both by matched filtering. That yields:
//! - the **delay** from "Windows handed the audio to the speaker" to "the mic heard it" (for
//!   Bluetooth this is mostly radio buffering), which our echo canceller needs as a starting point;
//! - how much that delay **varies** between sweeps (drift/jitter);
//! - the **echo level** relative to the room's noise, i.e. how hard the canceller has to work.
//!
//! The analysis functions are pure and tested on synthetic signals; [`run`] does the I/O.

use std::{
    f64::consts::PI,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::Context;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::wasapi::{self, Capture, CaptureMode, Endpoint, Render, Timeline};

pub const SWEEPS: usize = 3;
const SWEEP_SECS: f64 = 1.0;
const GAP_SECS: f64 = 0.8;
const LEAD_SECS: f64 = 1.0;
const TAIL_SECS: f64 = 1.0;
const AMPLITUDE: f32 = 0.5;

/// Exponential sine sweep `f0`→`f1` Hz with 10 ms raised-cosine fades.
pub fn sweep(rate: u32, secs: f64, f0: f64, f1: f64) -> Vec<f32> {
    let n = (secs * rate as f64) as usize;
    let k = (f1 / f0).ln();
    let fade = (rate as usize / 100).max(1);
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let phase = 2.0 * PI * f0 * secs / k * ((t / secs * k).exp() - 1.0);
            let edge = i.min(n - 1 - i);
            let gain = if edge < fade {
                0.5 - 0.5 * (PI * edge as f64 / fade as f64).cos()
            } else {
                1.0
            };
            (phase.sin() * gain) as f32
        })
        .collect()
}

/// Cross-correlation of `signal` with `template` at every lag ≥ 0 (FFT-based).
pub fn matched_filter(signal: &[f32], template: &[f32]) -> Vec<f32> {
    let n = (signal.len() + template.len()).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let to_c = |x: &[f32]| {
        let mut v: Vec<Complex<f32>> = x.iter().map(|&r| Complex::new(r, 0.0)).collect();
        v.resize(n, Complex::new(0.0, 0.0));
        v
    };
    let (mut a, mut b) = (to_c(signal), to_c(template));
    fwd.process(&mut a);
    fwd.process(&mut b);
    for (x, y) in a.iter_mut().zip(&b) {
        *x *= y.conj();
    }
    inv.process(&mut a);
    a.truncate(signal.len());
    a.into_iter().map(|c| c.re / n as f32).collect()
}

/// One sweep located in a recording.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Sample index where the sweep starts.
    pub index: usize,
    /// Peak height relative to the median correlation level, in dB.
    pub snr_db: f64,
}

/// Finds the strongest sweep start within each of `count` windows of `period` samples,
/// starting at `first_window`.
pub fn find_sweeps(
    signal: &[f32],
    template: &[f32],
    first_window: usize,
    period: usize,
    count: usize,
) -> Vec<Hit> {
    let mf = matched_filter(signal, template);
    let mut mags: Vec<f32> = mf.iter().map(|v| v.abs()).collect();
    let floor = {
        let mut sorted = mags.clone();
        sorted.sort_by(f32::total_cmp);
        sorted[sorted.len() / 2].max(1e-12)
    };
    (0..count)
        .filter_map(|k| {
            let from = first_window + k * period;
            let to = (from + period).min(mags.len());
            if from >= to {
                return None;
            }
            let (offset, peak) = mags[from..to]
                .iter_mut()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, v)| (i, *v))?;
            Some(Hit {
                index: from + offset,
                snr_db: 20.0 * (peak as f64 / floor as f64).log10(),
            })
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub speakers: String,
    pub microphone: String,
    pub measured_at_unix: u64,
    /// Median loopback→mic delay over the sweeps, ms.
    pub delay_ms: f64,
    /// Max minus min delay across sweeps, ms.
    pub delay_spread_ms: f64,
    pub delays_ms: Vec<f64>,
    /// Mic level with nothing playing.
    pub noise_floor_dbfs: f64,
    /// Mic level while the sweeps play.
    pub echo_dbfs: f64,
    /// Matched-filter peak above its median: how clearly the echo stands out, dB.
    pub echo_snr_db: f64,
    pub mic_discontinuities: u32,
    pub loopback_discontinuities: u32,
}

impl Calibration {
    /// Plain-language judgement used in reports and, later, by Grace herself.
    pub fn verdict(&self) -> Vec<String> {
        let mut notes = Vec::new();
        if self.echo_snr_db < 15.0 {
            notes.push(
                "the sweeps were hard to find in the mic: turn the speakers up and rerun".into(),
            );
        }
        if self.delay_spread_ms > 20.0 {
            notes.push(format!(
                "delay varies by {:.0} ms between sweeps: the echo canceller must track it",
                self.delay_spread_ms
            ));
        }
        if self.delay_ms > 120.0 {
            notes.push(format!(
                "{:.0} ms between playback and the mic: typical of Bluetooth, and it adds to every reply",
                self.delay_ms
            ));
        }
        if self.mic_discontinuities + self.loopback_discontinuities > 0 {
            notes.push("audio glitches were detected during the run; rerun to confirm".into());
        }
        notes
    }
}

/// Raw material of a calibration run, kept for offline echo-canceller work.
pub struct Recording {
    pub mic: Vec<f32>,
    pub mic_timeline: Timeline,
    pub loopback: Vec<f32>,
    pub loopback_timeline: Timeline,
    pub rate: u32,
}

fn capture_thread(
    ep: Endpoint,
    mode: CaptureMode,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<anyhow::Result<(Vec<f32>, Timeline)>> {
    thread::spawn(move || {
        wasapi::com_init()?;
        let mut cap = Capture::open(&ep, &mode)?;
        let mut samples = Vec::new();
        cap.start()?;
        while !stop.load(Ordering::Relaxed) {
            cap.read_mono(&mut samples)?;
            thread::sleep(Duration::from_millis(5));
        }
        cap.read_mono(&mut samples)?;
        Ok((samples, cap.timeline().clone()))
    })
}

fn dbfs(x: &[f32]) -> f64 {
    let p = x.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * p.max(1e-12).log10()
}

/// Plays the sweeps on `speakers` and measures them on `mic`. Makes sound for about 6 s.
pub fn run(mic: &Endpoint, speakers: &Endpoint) -> anyhow::Result<(Calibration, Recording)> {
    wasapi::com_init()?;
    let mut render = Render::open(speakers)?;
    let rate = render.rate();
    let template = sweep(rate, SWEEP_SECS, 100.0, 8000.0);
    let secs = |s: f64| (s * rate as f64) as usize;
    let period = secs(SWEEP_SECS + GAP_SECS);

    // Silence around the sweeps keeps the loopback stream delivering packets throughout.
    let mut pcm = vec![0.0f32; secs(LEAD_SECS)];
    for _ in 0..SWEEPS {
        pcm.extend(template.iter().map(|s| s * AMPLITUDE));
        pcm.extend(std::iter::repeat_n(0.0, secs(GAP_SECS)));
    }
    pcm.extend(std::iter::repeat_n(0.0, secs(TAIL_SECS)));

    let stop = Arc::new(AtomicBool::new(false));
    let mic_t = capture_thread(mic.clone(), CaptureMode::Raw, stop.clone());
    let lp_t = capture_thread(speakers.clone(), CaptureMode::Loopback, stop.clone());
    thread::sleep(Duration::from_millis(300)); // streams running before the first sample plays
    render.play_blocking(&pcm)?;
    thread::sleep(Duration::from_millis(600)); // Bluetooth tail
    stop.store(true, Ordering::Relaxed);
    let (mic_pcm, mic_tl) = mic_t
        .join()
        .expect("mic thread panicked")
        .context("mic capture")?;
    let (lp_pcm, lp_tl) = lp_t
        .join()
        .expect("loopback thread panicked")
        .context("loopback capture")?;
    anyhow::ensure!(
        mic_tl.rate == rate && lp_tl.rate == rate,
        "sample rates differ: mic {} Hz, loopback {} Hz, render {rate} Hz",
        mic_tl.rate,
        lp_tl.rate
    );

    // Loopback: sweeps sit at known positions after its first packet, give or take.
    let lp_hits = find_sweeps(&lp_pcm, &template, secs(LEAD_SECS / 2.0), period, SWEEPS);
    anyhow::ensure!(
        lp_hits.len() == SWEEPS,
        "could not find the sweeps in the loopback"
    );

    // Mic: search where each sweep can be, from its loopback time onwards (up to 1.5 s later).
    let lp_time = |i: usize| {
        lp_tl
            .time_of(i as f64)
            .context("loopback has no timestamps")
    };
    let mic_index_at = |t: f64| -> anyhow::Result<usize> {
        let t0 = mic_tl.time_of(0.0).context("mic has no timestamps")?;
        Ok(((t - t0) * rate as f64).max(0.0) as usize)
    };
    let mut delays = Vec::new();
    let mut snrs = Vec::new();
    for hit in &lp_hits {
        let t = lp_time(hit.index)?;
        let from = mic_index_at(t - 0.05)?;
        let window = secs(1.5);
        let end = (from + window + template.len()).min(mic_pcm.len());
        anyhow::ensure!(from < end, "mic recording ended before the sweeps");
        let found = find_sweeps(&mic_pcm[from..end], &template, 0, window, 1);
        let m = found.first().context("sweep not found in mic")?;
        let t_mic = mic_tl
            .time_of((from + m.index) as f64)
            .context("mic timestamps")?;
        delays.push((t_mic - t) * 1000.0);
        snrs.push(m.snr_db);
    }
    let mut sorted = delays.clone();
    sorted.sort_by(f64::total_cmp);
    let delay_ms = sorted[sorted.len() / 2];
    let spread = sorted.last().unwrap_or(&0.0) - sorted.first().unwrap_or(&0.0);

    let quiet_end = mic_index_at(lp_time(lp_hits[0].index)?)?;
    let noise = &mic_pcm[secs(0.2).min(quiet_end)..quiet_end];
    let sweeps_from = mic_index_at(lp_time(lp_hits[0].index)? + delay_ms / 1000.0)?;
    let sweep_part =
        &mic_pcm[sweeps_from.min(mic_pcm.len())..(sweeps_from + template.len()).min(mic_pcm.len())];

    let calibration = Calibration {
        speakers: speakers.name.clone(),
        microphone: mic.name.clone(),
        measured_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        delay_ms,
        delay_spread_ms: spread,
        delays_ms: delays,
        noise_floor_dbfs: dbfs(noise),
        echo_dbfs: dbfs(sweep_part),
        echo_snr_db: snrs.iter().copied().fold(f64::INFINITY, f64::min),
        mic_discontinuities: mic_tl.discontinuities,
        loopback_discontinuities: lp_tl.discontinuities,
    };
    let recording = Recording {
        mic: mic_pcm,
        mic_timeline: mic_tl,
        loopback: lp_pcm,
        loopback_timeline: lp_tl,
        rate,
    };
    Ok((calibration, recording))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x % 20_000) as f32 / 10_000.0 - 1.0) * amp
            })
            .collect()
    }

    #[test]
    fn finds_delayed_sweeps_in_noise() {
        let rate = 16_000;
        let t = sweep(rate, 0.5, 100.0, 6000.0);
        let period = 12_000;
        let delay = 3_456;
        let mut sig = noise(delay + period * 3 + 8_000, 0.3, 7);
        for k in 0..3 {
            // Attenuated, and a bit of "reverb" (a delayed copy).
            for (i, s) in t.iter().enumerate() {
                sig[delay + k * period + i] += s * 0.1;
                sig[delay + k * period + i + 200] += s * 0.03;
            }
        }
        let hits = find_sweeps(&sig, &t, 0, period, 3);
        assert_eq!(hits.len(), 3);
        for (k, h) in hits.iter().enumerate() {
            assert_eq!(h.index, delay + k * period, "sweep {k}");
            assert!(h.snr_db > 15.0, "snr {}", h.snr_db);
        }
    }

    #[test]
    fn sweep_is_bounded_and_faded() {
        let s = sweep(48_000, 1.0, 100.0, 8000.0);
        assert_eq!(s.len(), 48_000);
        assert!(s.iter().all(|v| v.abs() <= 1.0));
        assert!(s[0].abs() < 1e-3 && s[s.len() - 1].abs() < 0.05);
    }

    #[test]
    fn timeline_interpolates_between_packets() {
        let tl = Timeline {
            rate: 1000,
            stamps: vec![(0, 10_000_000), (100, 11_000_000)],
            discontinuities: 0,
        };
        assert!((tl.time_of(0.0).unwrap() - 1.0).abs() < 1e-9);
        assert!((tl.time_of(50.0).unwrap() - 1.05).abs() < 1e-9);
        assert!((tl.time_of(150.0).unwrap() - 1.15).abs() < 1e-9);
    }
}
