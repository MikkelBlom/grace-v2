//! Audio threads: three capture streams (ADR 0012) and one always-on render stream.
//!
//! Each capture thread sends `(source, capture time, mono samples)` chunks to the frontend.
//! The render thread keeps the speaker stream running and writes silence when Grace is quiet:
//! that keeps a Bluetooth speaker awake (no wake-up delay before she speaks) and keeps the
//! loopback reference continuous for echo cancellation.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use anyhow::Context;
use grace_audio::wasapi::{self, Capture, CaptureMode, Endpoint, Render};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Raw mic: input to our AEC (decides *when* someone speaks).
    Mic,
    /// Windows communications-mode mic: input to STT (hears *what* was said).
    Comms,
    /// What the speakers play: the AEC reference.
    Loopback,
}

pub struct Chunk {
    pub source: Source,
    pub time: f64,
    pub samples: Vec<f32>,
}

pub const RATE: u32 = 48_000;
/// Samples quieter than this (about -54 dBFS) count as silence for "is Grace still talking".
const AUDIBLE: f32 = 0.002;

pub fn spawn_capture(
    source: Source,
    ep: Endpoint,
    mode: CaptureMode,
    tx: mpsc::Sender<Chunk>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<anyhow::Result<()>> {
    thread::Builder::new()
        .name(format!("capture-{source:?}"))
        .spawn(move || {
            wasapi::com_init()?;
            let mut cap =
                Capture::open(&ep, &mode).with_context(|| format!("opening {source:?} capture"))?;
            anyhow::ensure!(
                cap.rate() == RATE,
                "{source:?} runs at {} Hz; the engine expects {RATE}",
                cap.rate()
            );
            cap.start()?;
            let mut total = 0u64;
            let mut buf = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                buf.clear();
                let n = cap.read_mono(&mut buf)?;
                if n > 0 {
                    let time = cap
                        .timeline()
                        .time_of(total as f64)
                        .unwrap_or_else(crate::clock::now);
                    total += n as u64;
                    if tx
                        .send(Chunk {
                            source,
                            time,
                            samples: buf.clone(),
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(3));
            }
            Ok(())
        })
        .expect("spawning capture thread")
}

pub enum RenderCmd {
    /// Mono samples at [`RATE`] for the given turn.
    Enqueue {
        turn: u64,
        samples: Vec<f32>,
    },
    Duck(bool),
    /// Stop now: fade out over ~20 ms and drop everything queued.
    Cancel,
}

#[derive(Debug, Clone, Copy)]
pub enum RenderEvent {
    /// The first sample of `turn` reached the device buffer.
    Started { turn: u64 },
    /// Everything queued for `turn` has been written and has drained from the device buffer.
    Drained { turn: u64, time: f64 },
}

pub fn spawn_render(
    ep: Endpoint,
    cmds: mpsc::Receiver<RenderCmd>,
    events: tokio::sync::mpsc::UnboundedSender<RenderEvent>,
    talking: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<anyhow::Result<()>> {
    thread::Builder::new()
        .name("render".into())
        .spawn(move || {
            wasapi::com_init()?;
            let mut out = Render::open(&ep)?;
            anyhow::ensure!(
                out.rate() == RATE,
                "speakers run at {} Hz; the engine expects {RATE}",
                out.rate()
            );
            out.start()?;
            let mut queue: VecDeque<(u64, f32)> = VecDeque::new();
            let (mut gain, mut target) = (1.0f32, 1.0f32);
            let mut fade_out = 0usize;
            let mut current: Option<u64> = None;
            let mut started = false;
            let mut scratch = Vec::new();
            let mut last_busy = 0.0f64;
            let mut drain_at: Option<f64> = None;
            let mut audible_left = 0usize;
            while !stop.load(Ordering::Relaxed) {
                while let Ok(cmd) = cmds.try_recv() {
                    match cmd {
                        RenderCmd::Enqueue { turn, samples } => {
                            if current != Some(turn) {
                                current = Some(turn);
                                started = false;
                            }
                            // Track where the last audible sample is: TTS ends replies with ~0.5 s
                            // of silence, and a reply is over once its voice is, not its padding.
                            if let Some(last) = samples.iter().rposition(|s| s.abs() > AUDIBLE) {
                                audible_left = queue.len() + last + 1;
                            }
                            queue.extend(samples.into_iter().map(|s| (turn, s)));
                        }
                        RenderCmd::Duck(on) => target = if on { 0.3 } else { 1.0 },
                        RenderCmd::Cancel => fade_out = (RATE / 50) as usize,
                    }
                }
                let n = out.writable()?.min(RATE as usize / 50);
                scratch.clear();
                for _ in 0..n {
                    gain += (target - gain) * 0.002; // ~10 ms smoothing at 48 kHz
                    let s = match queue
                        .pop_front()
                        .inspect(|_| audible_left = audible_left.saturating_sub(1))
                    {
                        Some((turn, s)) => {
                            if !started && Some(turn) == current {
                                started = true;
                                let _ = events.send(RenderEvent::Started { turn });
                            }
                            if fade_out > 0 {
                                fade_out -= 1;
                                let f = fade_out as f32 / (RATE / 50) as f32;
                                if fade_out == 0 {
                                    queue.clear();
                                    audible_left = 0;
                                }
                                s * gain * f
                            } else {
                                s * gain
                            }
                        }
                        None => {
                            fade_out = 0; // a cancel with nothing queued must not mute the next reply
                            0.0
                        }
                    };
                    scratch.push(s);
                }
                out.write(&scratch)?;
                let busy = audible_left > 0;
                // Her voice is still in the room for the device buffer + speaker delay (Bluetooth
                // ~150 ms) after the last sample leaves the queue.
                if busy {
                    last_busy = crate::clock::now();
                }
                talking.store(
                    busy || crate::clock::now() - last_busy < 0.5,
                    Ordering::Relaxed,
                );
                // The device buffer is kept full of silence (see module docs), so "drained" means:
                // the turn's last sample was written, plus the time the buffer ahead of it takes.
                if busy {
                    drain_at = None;
                } else if let Some(turn) = current
                    && started
                {
                    let now = crate::clock::now();
                    match drain_at {
                        None => drain_at = Some(now + out.queued()? as f64 / RATE as f64),
                        Some(at) if now >= at => {
                            let _ = events.send(RenderEvent::Drained { turn, time: now });
                            current = None;
                            drain_at = None;
                        }
                        Some(_) => {}
                    }
                }
                thread::sleep(Duration::from_millis(4));
            }
            Ok(())
        })
        .expect("spawning render thread")
}
