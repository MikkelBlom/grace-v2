//! Audio threads: two capture streams (raw mic, speaker loopback) and one always-on render stream.
//!
//! Each capture thread sends `(source, capture time, mono samples)` chunks to the frontend.
//! The render thread keeps the speaker stream running and writes silence when Grace is quiet:
//! that keeps a Bluetooth device awake (no wake-up delay before she speaks) and keeps the
//! loopback reference continuous for echo cancellation.
//!
//! Every thread heals itself: when a device errors (Bluetooth sleeping or reconnecting, the
//! default device changing, the Windows audio service restarting) it logs the error to the trace
//! and reopens the device every second, instead of dying. A silently dead mic thread made Grace
//! deaf for two minutes in a live run.

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
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Raw mic: input to our AEC, then VAD and STT.
    Mic,
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
const RETRY: Duration = Duration::from_secs(1);

/// Runs `body` until `stop`, reopening after errors. `body` returns Ok when asked to stop.
fn self_healing(name: String, stop: Arc<AtomicBool>, mut body: impl FnMut() -> anyhow::Result<()>) {
    let mut failures = 0u32;
    while !stop.load(Ordering::Relaxed) {
        match body() {
            Ok(()) => return,
            Err(e) => {
                failures += 1;
                // First failure loudly (console + trace), repeats only to the trace.
                if failures == 1 {
                    eprintln!("{name}: {e:#}; retrying every second");
                }
                warn!(thread = %name, error = %format!("{e:#}"), failures, "audio device error, reopening");
                thread::sleep(RETRY);
            }
        }
    }
}

pub fn spawn_capture(
    source: Source,
    ep: Endpoint,
    mode: CaptureMode,
    tx: mpsc::Sender<Chunk>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    let name = format!("capture-{source:?}");
    thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            if let Err(e) = wasapi::com_init() {
                eprintln!("{name}: COM init failed: {e:#}");
                return;
            }
            let mut opened_before = false;
            self_healing(name.clone(), stop.clone(), || {
                let mut cap = Capture::open(&ep, &mode)
                    .with_context(|| format!("opening {source:?} capture"))?;
                anyhow::ensure!(
                    cap.rate() == RATE,
                    "{source:?} runs at {} Hz; the engine expects {RATE}",
                    cap.rate()
                );
                cap.start()?;
                if opened_before {
                    info!(thread = %name, "audio device reopened");
                    eprintln!("{name}: reopened");
                }
                opened_before = true;
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
                            return Ok(());
                        }
                    }
                    thread::sleep(Duration::from_millis(3));
                }
                Ok(())
            });
        })
        .expect("spawning capture thread")
}

pub enum RenderCmd {
    /// Mono samples at [`RATE`] for the given turn.
    Enqueue {
        turn: u64,
        samples: Vec<f32>,
    },
    /// All of `turn`'s audio has been enqueued: once it has played, the reply is over.
    EndOfTurn {
        turn: u64,
    },
    Duck(bool),
    /// Stop now: fade out over ~20 ms and drop everything queued.
    Cancel,
}

#[derive(Debug, Clone, Copy)]
pub enum RenderEvent {
    /// The first sample of `turn` reached the device buffer.
    Started { turn: u64 },
    /// `turn`'s last audible sample has played (its EndOfTurn was received first).
    Drained { turn: u64, time: f64 },
}

/// Playback state that survives a device reopen.
struct Playback {
    queue: VecDeque<(u64, f32)>,
    gain: f32,
    target: f32,
    fade_out: usize,
    current: Option<u64>,
    started: bool,
    /// EndOfTurn received for `current`: until then a momentarily empty queue (TTS slower than
    /// playback between sentences) does not mean the reply is over.
    ended: bool,
    audible_left: usize,
    last_busy: f64,
    drain_at: Option<f64>,
}

pub fn spawn_render(
    ep: Endpoint,
    cmds: mpsc::Receiver<RenderCmd>,
    events: tokio::sync::mpsc::UnboundedSender<RenderEvent>,
    talking: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name("render".into())
        .spawn(move || {
            if let Err(e) = wasapi::com_init() {
                eprintln!("render: COM init failed: {e:#}");
                return;
            }
            let mut p = Playback {
                queue: VecDeque::new(),
                gain: 1.0,
                target: 1.0,
                fade_out: 0,
                current: None,
                started: false,
                ended: false,
                audible_left: 0,
                last_busy: 0.0,
                drain_at: None,
            };
            let mut opened_before = false;
            self_healing("render".into(), stop.clone(), || {
                let mut out = Render::open(&ep)?;
                anyhow::ensure!(
                    out.rate() == RATE,
                    "speakers run at {} Hz; the engine expects {RATE}",
                    out.rate()
                );
                out.start()?;
                if opened_before {
                    info!(thread = "render", "audio device reopened");
                    eprintln!("render: reopened");
                }
                opened_before = true;
                let mut scratch = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    while let Ok(cmd) = cmds.try_recv() {
                        p.apply(cmd);
                    }
                    let n = out.writable()?.min(RATE as usize / 50);
                    scratch.clear();
                    for _ in 0..n {
                        scratch.push(p.next_sample(&events));
                    }
                    out.write(&scratch)?;
                    p.update_state(out.queued()?, &events, &talking);
                    thread::sleep(Duration::from_millis(4));
                }
                Ok(())
            });
        })
        .expect("spawning render thread")
}

impl Playback {
    fn apply(&mut self, cmd: RenderCmd) {
        match cmd {
            RenderCmd::Enqueue { turn, samples } => {
                if self.current != Some(turn) {
                    self.current = Some(turn);
                    self.started = false;
                    self.ended = false;
                }
                // Track where the last audible sample is: TTS ends replies with ~0.5 s of
                // silence, and a reply is over once its voice is, not its padding.
                if let Some(last) = samples.iter().rposition(|s| s.abs() > AUDIBLE) {
                    self.audible_left = self.queue.len() + last + 1;
                }
                self.queue.extend(samples.into_iter().map(|s| (turn, s)));
            }
            RenderCmd::EndOfTurn { turn } => {
                if self.current == Some(turn) {
                    self.ended = true;
                }
            }
            RenderCmd::Duck(on) => self.target = if on { 0.3 } else { 1.0 },
            RenderCmd::Cancel => self.fade_out = (RATE / 50) as usize,
        }
    }

    fn next_sample(&mut self, events: &tokio::sync::mpsc::UnboundedSender<RenderEvent>) -> f32 {
        self.gain += (self.target - self.gain) * 0.002; // ~10 ms smoothing at 48 kHz
        match self.queue.pop_front() {
            Some((turn, s)) => {
                self.audible_left = self.audible_left.saturating_sub(1);
                if !self.started && Some(turn) == self.current {
                    self.started = true;
                    let _ = events.send(RenderEvent::Started { turn });
                }
                if self.fade_out > 0 {
                    self.fade_out -= 1;
                    let f = self.fade_out as f32 / (RATE / 50) as f32;
                    if self.fade_out == 0 {
                        self.queue.clear();
                        self.audible_left = 0;
                    }
                    s * self.gain * f
                } else {
                    s * self.gain
                }
            }
            None => {
                self.fade_out = 0; // a cancel with nothing queued must not mute the next reply
                0.0
            }
        }
    }

    fn update_state(
        &mut self,
        device_queued: usize,
        events: &tokio::sync::mpsc::UnboundedSender<RenderEvent>,
        talking: &AtomicBool,
    ) {
        let now = crate::clock::now();
        let busy = self.audible_left > 0;
        // Her voice is still in the room for the device buffer + speaker delay (Bluetooth
        // ~150 ms) after the last sample leaves the queue.
        if busy {
            self.last_busy = now;
        }
        talking.store(busy || now - self.last_busy < 0.5, Ordering::Relaxed);
        // The device buffer is kept full of silence, so "drained" means: the reply is complete,
        // its last audible sample was written, and the device buffer ahead of it has played.
        if busy || !self.ended {
            self.drain_at = None;
        } else if let Some(turn) = self.current
            && self.started
        {
            match self.drain_at {
                None => self.drain_at = Some(now + device_queued as f64 / RATE as f64),
                Some(at) if now >= at => {
                    let _ = events.send(RenderEvent::Drained { turn, time: now });
                    self.current = None;
                    self.drain_at = None;
                }
                Some(_) => {}
            }
        }
    }
}
