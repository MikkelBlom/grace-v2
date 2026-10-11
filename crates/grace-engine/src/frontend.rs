//! The real-time audio path: align → AEC3 → 16 kHz → VAD → gate → speech edges.
//!
//! - Raw mic frames (10 ms at 48 kHz) are echo-cancelled against the loopback recorded at the
//!   same clock time, shifted by the tracked speaker delay minus a small lead (ADR 0006).
//! - AEC3 gives two signals: the full chain (linear filter + residual echo suppression + noise
//!   suppression), which feeds VAD and Smart Turn, and the linear-filter-only output, which is
//!   kept for STT (ADR 0014: suppression artifacts cost recognition accuracy).
//! - Every 2 s, while something is playing, the delay is re-estimated by GCC-PHAT.
//!
//! [`Frontend`] is pure processing driven by timestamped chunks, so `grace-sim` runs the exact
//! same code faster than real time on recorded or synthetic scenes; [`spawn`] wraps it in the
//! live engine's thread.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use aec3::{nodes::audio::AudioFormat, pipelines::linear};
use grace_core::vad_gate::{VadEdge, VadGate, VadGateConfig};
use grace_vad::{Decimate3, FRAME as VAD_FRAME, SileroVad};

use crate::{
    audio_io::{Chunk, RATE, Source},
    delay::{DelayTracker, gcc_phat},
    timed::TimedRing,
};

const FRAME: usize = (RATE / 100) as usize; // 10 ms at 48 kHz
const VAD_SECS: f64 = VAD_FRAME as f64 / 16_000.0;

/// Audio the conductor reads for Smart Turn and STT.
pub struct Shared {
    /// Echo-cancelled mic after the full AEC3 chain, 16 kHz.
    pub aec16: TimedRing,
    /// Echo-cancelled mic after AEC3's linear filter only (no suppression), 16 kHz.
    pub lin16: TimedRing,
    /// What the speakers played, 16 kHz (to know whether a turn overlapped playback).
    pub speakers16: TimedRing,
    /// Raw mic, 16 kHz (saved per turn for diagnosis).
    pub raw16: TimedRing,
}

impl Shared {
    pub fn new(seconds: f64) -> Self {
        Self {
            aec16: TimedRing::new(16_000, seconds),
            lin16: TimedRing::new(16_000, seconds),
            speakers16: TimedRing::new(16_000, seconds),
            raw16: TimedRing::new(16_000, seconds),
        }
    }

    /// Moves what the frontend produced for one chunk into the rings.
    pub fn absorb(&mut self, out: &mut Produced) {
        for (ring, parts) in [
            (&mut self.aec16, &mut out.aec16),
            (&mut self.lin16, &mut out.lin16),
            (&mut self.speakers16, &mut out.speakers16),
            (&mut self.raw16, &mut out.raw16),
        ] {
            for (t, samples) in parts.drain(..) {
                ring.push(t, &samples);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FrontEvent {
    SpeechStarted { time: f64 },
    SpeechStopped { time: f64 },
    Delay { ms: f64, confidence: f32 },
}

pub struct FrontendConfig {
    pub vad_model: std::path::PathBuf,
    pub initial_delay_ms: f64,
    pub lead_ms: f64,
    pub gate: VadGateConfig,
    /// AEC3's noise suppressor in the full chain (VAD and Smart Turn input).
    pub noise_suppression: bool,
}

/// What one chunk produced: timestamped 16 kHz pieces for the rings, plus speech/delay events.
#[derive(Default)]
pub struct Produced {
    pub aec16: Vec<(f64, Vec<f32>)>,
    pub lin16: Vec<(f64, Vec<f32>)>,
    pub speakers16: Vec<(f64, Vec<f32>)>,
    pub raw16: Vec<(f64, Vec<f32>)>,
    pub events: Vec<FrontEvent>,
}

pub struct Frontend {
    aec: linear::LinearPipeline,
    vad: SileroVad,
    gate: VadGate,
    tracker: DelayTracker,
    lead_ms: f64,
    lp48: TimedRing,
    lp16: TimedRing,
    mic16: TimedRing,
    dec_aec: Decimate3,
    dec_mic: Decimate3,
    dec_lp: Decimate3,
    pending: Vec<f32>,
    pending_time: f64,
    vad_buf: Vec<f32>,
    /// Time of VAD frame 0.
    vad_t0: Option<f64>,
    out: Vec<f32>,
    next_delay_check: f64,
}

impl Frontend {
    pub fn new(cfg: FrontendConfig) -> anyhow::Result<Self> {
        let format = AudioFormat::ten_ms(RATE, 1);
        let aec = linear::builder(format, format)
            .initial_delay_ms(cfg.lead_ms as i32)
            .enable_gain_controller2(false)
            .enable_noise_suppression(cfg.noise_suppression)
            .export_linear_output(true)
            .build()
            .map_err(|e| anyhow::anyhow!("AEC3: {e}"))?;
        Ok(Self {
            aec,
            vad: SileroVad::new(&cfg.vad_model)?,
            gate: VadGate::new(cfg.gate),
            tracker: DelayTracker::new(cfg.initial_delay_ms),
            lead_ms: cfg.lead_ms,
            lp48: TimedRing::new(RATE, 3.0),
            lp16: TimedRing::new(16_000, 6.0),
            mic16: TimedRing::new(16_000, 6.0),
            dec_aec: Decimate3::default(),
            dec_mic: Decimate3::default(),
            dec_lp: Decimate3::default(),
            pending: Vec::new(),
            pending_time: 0.0,
            vad_buf: Vec::new(),
            vad_t0: None,
            out: vec![0.0; FRAME],
            next_delay_check: 0.0,
        })
    }

    /// Current speaker delay estimate in ms.
    pub fn delay_ms(&self) -> f64 {
        self.tracker.current_ms
    }

    /// Processes one capture chunk. `talking` is whether Grace is (still) audible.
    pub fn push(&mut self, chunk: &Chunk, talking: bool, out: &mut Produced) -> anyhow::Result<()> {
        let mut tmp = Vec::new();
        match chunk.source {
            Source::Loopback => {
                self.lp48.push(chunk.time, &chunk.samples);
                self.dec_lp.push(&chunk.samples, &mut tmp);
                self.lp16.push(chunk.time, &tmp);
                out.speakers16.push((chunk.time, tmp));
            }
            Source::Mic => {
                if self.pending.is_empty() {
                    self.pending_time = chunk.time;
                }
                self.pending.extend_from_slice(&chunk.samples);
                self.dec_mic.push(&chunk.samples, &mut tmp);
                self.mic16.push(chunk.time, &tmp);
                out.raw16.push((chunk.time, tmp));
                while self.pending.len() >= FRAME {
                    let frame: Vec<f32> = self.pending.drain(..FRAME).collect();
                    let t = self.pending_time;
                    self.pending_time += FRAME as f64 / RATE as f64;
                    self.frame(t, &frame, talking, out)?;
                }
                let now = self.mic16.end();
                if now >= self.next_delay_check {
                    self.next_delay_check = now + 2.0;
                    if let Some((ms, conf)) = estimate_delay(&self.lp16, &self.mic16, now)
                        && let Some(new) = self.tracker.update(ms, conf)
                    {
                        out.events.push(FrontEvent::Delay {
                            ms: new,
                            confidence: conf,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn frame(
        &mut self,
        t: f64,
        frame: &[f32],
        talking: bool,
        out: &mut Produced,
    ) -> anyhow::Result<()> {
        // Reference = what the speakers played (delay - lead) before this mic frame.
        let reference = self
            .lp48
            .read(t - (self.tracker.current_ms - self.lead_ms) / 1000.0, FRAME);
        self.aec
            .handle_render_frame(&reference)
            .map_err(|e| anyhow::anyhow!("AEC3 render: {e}"))?;
        let ready = self
            .aec
            .process_capture_frame(frame, &mut self.out)
            .map_err(|e| anyhow::anyhow!("AEC3 capture: {e}"))?;
        // The linear output is already 16 kHz; it is produced in step with the main output.
        while let Some(p) = self
            .aec
            .try_pull_linear_output()
            .map_err(|e| anyhow::anyhow!("AEC3 linear: {e}"))?
        {
            out.lin16.push((t, p.payload().samples().to_vec()));
        }
        if !ready {
            return Ok(());
        }
        let mut tmp = Vec::new();
        self.dec_aec.push(&self.out, &mut tmp);
        if self.vad_t0.is_none() {
            self.vad_t0 = Some(t);
        }
        self.vad_buf.extend_from_slice(&tmp);
        out.aec16.push((t, tmp));
        self.gate.set_grace_talking(talking);
        while self.vad_buf.len() >= VAD_FRAME {
            let f: Vec<f32> = self.vad_buf.drain(..VAD_FRAME).collect();
            let p = self.vad.process(&f)?;
            let t0 = self.vad_t0.unwrap_or(t);
            let at = |frame: u64| t0 + frame as f64 * VAD_SECS;
            match self.gate.push(p) {
                Some(VadEdge::Started { frame }) => {
                    out.events
                        .push(FrontEvent::SpeechStarted { time: at(frame) });
                }
                Some(VadEdge::Stopped { frame }) => {
                    out.events
                        .push(FrontEvent::SpeechStopped { time: at(frame) });
                }
                None => {}
            }
        }
        Ok(())
    }
}

pub fn spawn(
    cfg: FrontendConfig,
    chunks: mpsc::Receiver<Chunk>,
    shared: Arc<Mutex<Shared>>,
    events: tokio::sync::mpsc::UnboundedSender<FrontEvent>,
    talking: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name("frontend".into())
        .spawn(move || {
            if let Err(e) = run(cfg, chunks, shared, events, talking, stop) {
                eprintln!("frontend: {e:#}");
                tracing::error!(error = %format!("{e:#}"), "frontend failed");
            }
        })
        .expect("spawning frontend thread")
}

fn run(
    cfg: FrontendConfig,
    chunks: mpsc::Receiver<Chunk>,
    shared: Arc<Mutex<Shared>>,
    events: tokio::sync::mpsc::UnboundedSender<FrontEvent>,
    talking: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let mut front = Frontend::new(cfg)?;
    let mut out = Produced::default();
    while !stop.load(Ordering::Relaxed) {
        let chunk = match chunks.recv_timeout(Duration::from_millis(50)) {
            Ok(c) => c,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        front.push(&chunk, talking.load(Ordering::Relaxed), &mut out)?;
        shared.lock().expect("shared poisoned").absorb(&mut out);
        for ev in out.events.drain(..) {
            let _ = events.send(ev);
        }
    }
    Ok(())
}

/// Delay estimate over the last 3 s, or None while the speakers are (nearly) silent.
fn estimate_delay(lp16: &TimedRing, mic16: &TimedRing, now: f64) -> Option<(f64, f32)> {
    let (t0, t1) = (now - 3.2, now - 0.2);
    let reference = lp16.read_span(t0, t1);
    let rms = (reference.iter().map(|v| v * v).sum::<f32>() / reference.len().max(1) as f32).sqrt();
    if rms < 0.003 {
        return None;
    }
    let mic = mic16.read_span(t0, t1);
    let (lag, conf) = gcc_phat(&reference, &mic, 8_000);
    Some((lag as f64 / 16.0, conf))
}
