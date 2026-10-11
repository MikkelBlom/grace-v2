//! The real-time audio path (one thread): align → AEC3 → 16 kHz → VAD → gate → speech edges.
//!
//! - Raw mic frames (10 ms at 48 kHz) are echo-cancelled against the loopback recorded at the
//!   same clock time, shifted by the tracked speaker delay minus a small lead (ADR 0006).
//! - The cancelled signal feeds VAD and is kept (16 kHz) for Smart Turn.
//! - STT also reads the echo-cancelled signal (ADR 0013).
//! - Every 2 s, while something is playing, the delay is re-estimated by GCC-PHAT.

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
    /// Echo-cancelled mic, 16 kHz.
    pub aec16: TimedRing,
    /// What the speakers played, 16 kHz (to know whether a turn overlapped playback).
    pub speakers16: TimedRing,
    /// Raw mic, 16 kHz (saved per turn for diagnosis).
    pub raw16: TimedRing,
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
    let format = AudioFormat::ten_ms(RATE, 1);
    let mut aec = linear::builder(format, format)
        .initial_delay_ms(cfg.lead_ms as i32)
        .enable_gain_controller2(false)
        .build()
        .map_err(|e| anyhow::anyhow!("AEC3: {e}"))?;
    let mut vad = SileroVad::new(&cfg.vad_model)?;
    let mut gate = VadGate::new(cfg.gate);
    let mut tracker = DelayTracker::new(cfg.initial_delay_ms);

    let mut lp48 = TimedRing::new(RATE, 3.0);
    let mut lp16 = TimedRing::new(16_000, 6.0);
    let mut mic16 = TimedRing::new(16_000, 6.0);
    let (mut dec_aec, mut dec_mic, mut dec_lp) = (
        Decimate3::default(),
        Decimate3::default(),
        Decimate3::default(),
    );

    let mut pending: Vec<f32> = Vec::new();
    let mut pending_time = 0.0;
    let mut vad_buf: Vec<f32> = Vec::new();
    let mut vad_t0: Option<f64> = None; // time of VAD frame 0
    let mut out = vec![0.0f32; FRAME];
    let mut next_delay_check = 0.0;
    let mut tmp = Vec::new();

    while !stop.load(Ordering::Relaxed) {
        let chunk = match chunks.recv_timeout(Duration::from_millis(50)) {
            Ok(c) => c,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match chunk.source {
            Source::Loopback => {
                lp48.push(chunk.time, &chunk.samples);
                tmp.clear();
                dec_lp.push(&chunk.samples, &mut tmp);
                lp16.push(chunk.time, &tmp);
                shared
                    .lock()
                    .expect("shared poisoned")
                    .speakers16
                    .push(chunk.time, &tmp);
            }
            Source::Mic => {
                if pending.is_empty() {
                    pending_time = chunk.time;
                }
                pending.extend_from_slice(&chunk.samples);
                tmp.clear();
                dec_mic.push(&chunk.samples, &mut tmp);
                mic16.push(chunk.time, &tmp);
                shared
                    .lock()
                    .expect("shared poisoned")
                    .raw16
                    .push(chunk.time, &tmp);
                while pending.len() >= FRAME {
                    let frame: Vec<f32> = pending.drain(..FRAME).collect();
                    let t = pending_time;
                    pending_time += FRAME as f64 / RATE as f64;
                    // Reference = what the speakers played (delay - lead) before this mic frame.
                    let reference =
                        lp48.read(t - (tracker.current_ms - cfg.lead_ms) / 1000.0, FRAME);
                    aec.handle_render_frame(&reference)
                        .map_err(|e| anyhow::anyhow!("AEC3 render: {e}"))?;
                    if !aec
                        .process_capture_frame(&frame, &mut out)
                        .map_err(|e| anyhow::anyhow!("AEC3 capture: {e}"))?
                    {
                        continue;
                    }
                    tmp.clear();
                    dec_aec.push(&out, &mut tmp);
                    shared.lock().expect("shared poisoned").aec16.push(t, &tmp);
                    if vad_t0.is_none() {
                        vad_t0 = Some(t);
                    }
                    vad_buf.extend_from_slice(&tmp);
                    gate.set_grace_talking(talking.load(Ordering::Relaxed));
                    while vad_buf.len() >= VAD_FRAME {
                        let f: Vec<f32> = vad_buf.drain(..VAD_FRAME).collect();
                        let p = vad.process(&f)?;
                        let at = |frame: u64| vad_t0.unwrap_or(t) + frame as f64 * VAD_SECS;
                        match gate.push(p) {
                            Some(VadEdge::Started { frame }) => {
                                let _ = events.send(FrontEvent::SpeechStarted { time: at(frame) });
                            }
                            Some(VadEdge::Stopped { frame }) => {
                                let _ = events.send(FrontEvent::SpeechStopped { time: at(frame) });
                            }
                            None => {}
                        }
                    }
                }
                let now = mic16.end();
                if now >= next_delay_check {
                    next_delay_check = now + 2.0;
                    if let Some((ms, conf)) = estimate_delay(&lp16, &mic16, now)
                        && let Some(new) = tracker.update(ms, conf)
                    {
                        let _ = events.send(FrontEvent::Delay {
                            ms: new,
                            confidence: conf,
                        });
                    }
                }
            }
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
