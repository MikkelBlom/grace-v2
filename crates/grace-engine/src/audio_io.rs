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
        atomic::{AtomicBool, AtomicU64, Ordering},
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

/// Commands for the render thread. `reply` ids are the conductor's, one per reply (a turn can
/// have several: a retracted one and its replacement).
pub enum RenderCmd {
    /// Mono samples at [`RATE`] for the given reply.
    Enqueue {
        reply: u64,
        samples: Vec<f32>,
    },
    /// All of `reply`'s audio has been enqueued: once it has played, the reply is over.
    EndOfTurn {
        reply: u64,
    },
    Duck(bool),
    /// Hold playback (silence) without dropping the queue; `false` carries on.
    Pause(bool),
    /// Stop the current reply now: fade out over ~20 ms and drop the rest of it. Audio of a
    /// reply enqueued after it is kept.
    Cancel,
}

#[derive(Debug, Clone, Copy)]
pub enum RenderEvent {
    /// The first sample of `reply` reached the device buffer.
    Started { reply: u64 },
    /// `reply`'s last audible sample has played (its EndOfTurn was received first).
    Drained { reply: u64, time: f64 },
}

/// How many samples of which reply have been played (sent to the device), shared with the
/// conductor so that a cut-off reply leaves in the conversation only what was heard.
#[derive(Clone, Default)]
pub struct Played(Arc<AtomicU64>);

impl Played {
    const MASK: u64 = (1 << 40) - 1;

    /// (reply, samples). One word holds both, so the pair is always consistent.
    pub fn get(&self) -> (u64, usize) {
        let v = self.0.load(Ordering::Relaxed);
        (v >> 40, (v & Self::MASK) as usize)
    }

    /// Samples of `reply` played so far (0 if another reply played since).
    pub fn of(&self, reply: u64) -> usize {
        match self.get() {
            (r, n) if r == reply => n,
            _ => 0,
        }
    }

    fn set(&self, (reply, samples): (u64, usize)) {
        self.0.store(
            (reply << 40) | (samples as u64 & Self::MASK),
            Ordering::Relaxed,
        );
    }
}

/// Playback state that survives a device reopen.
struct Playback {
    queue: VecDeque<(u64, f32)>,
    gain: f32,
    target: f32,
    current: Option<u64>,
    started: bool,
    /// EndOfTurn received for `current`: until then a momentarily empty queue (TTS slower than
    /// playback between sentences) does not mean the reply is over.
    ended: bool,
    audible_left: usize,
    last_busy: f64,
    drain_at: Option<f64>,
    paused: bool,
    /// Samples left in the fade-out before a pause takes hold (cutting mid-waveform clicks).
    pausing: usize,
    /// The reply whose samples were played last, and how many of them.
    played: (u64, usize),
}

/// 10 ms: long enough not to click, short enough that Grace still stops at once.
const PAUSE_FADE: usize = (RATE / 100) as usize;
const CANCEL_FADE: usize = (RATE / 50) as usize;

pub fn spawn_render(
    ep: Endpoint,
    cmds: mpsc::Receiver<RenderCmd>,
    events: tokio::sync::mpsc::UnboundedSender<RenderEvent>,
    talking: Arc<AtomicBool>,
    played: Played,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name("render".into())
        .spawn(move || {
            if let Err(e) = wasapi::com_init() {
                eprintln!("render: COM init failed: {e:#}");
                return;
            }
            let mut p = Playback::new();
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
                    played.set(p.played);
                    p.update_state(out.queued()?, &events, &talking);
                    thread::sleep(Duration::from_millis(4));
                }
                Ok(())
            });
        })
        .expect("spawning render thread")
}

impl Playback {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            gain: 1.0,
            target: 1.0,
            current: None,
            started: false,
            ended: false,
            audible_left: 0,
            last_busy: 0.0,
            drain_at: None,
            paused: false,
            pausing: 0,
            played: (0, 0),
        }
    }

    fn apply(&mut self, cmd: RenderCmd) {
        match cmd {
            RenderCmd::Enqueue {
                reply: turn,
                samples,
            } => {
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
            RenderCmd::EndOfTurn { reply: turn } => {
                if self.current == Some(turn) {
                    self.ended = true;
                }
            }
            RenderCmd::Duck(on) => self.target = if on { 0.3 } else { 1.0 },
            RenderCmd::Pause(true) => {
                if !self.paused && self.pausing == 0 {
                    // Fade out first; an empty queue has nothing to click.
                    self.pausing = PAUSE_FADE;
                    self.paused = self.queue.is_empty();
                }
            }
            RenderCmd::Pause(false) => {
                if self.paused {
                    self.gain = 0.0; // fade back in rather than click
                }
                self.paused = false;
                self.pausing = 0;
            }
            RenderCmd::Cancel => {
                let Some(cut) = self.current.take() else {
                    return;
                };
                (self.started, self.ended, self.drain_at) = (false, false, None);
                // A paused reply is already silent; a playing one fades over its next 20 ms.
                let fade = if self.paused || self.pausing > 0 {
                    0
                } else {
                    CANCEL_FADE
                };
                (self.paused, self.pausing) = (false, 0);
                let mut head = Vec::with_capacity(fade);
                while head.len() < fade {
                    match self.queue.front() {
                        Some(&(r, s)) if r == cut => {
                            self.queue.pop_front();
                            head.push(s);
                        }
                        _ => break,
                    }
                }
                self.queue.retain(|(r, _)| *r != cut);
                let n = head.len() as f32;
                for (i, s) in head.into_iter().enumerate().rev() {
                    self.queue.push_front((cut, s * (n - i as f32) / n));
                }
                self.audible_left = self
                    .queue
                    .iter()
                    .rposition(|(_, s)| s.abs() > AUDIBLE)
                    .map_or(0, |i| i + 1);
            }
        }
    }

    fn next_sample(&mut self, events: &tokio::sync::mpsc::UnboundedSender<RenderEvent>) -> f32 {
        self.gain += (self.target - self.gain) * 0.002; // ~10 ms smoothing at 48 kHz
        if self.paused {
            return 0.0;
        }
        match self.queue.pop_front() {
            Some((turn, s)) => {
                self.audible_left = self.audible_left.saturating_sub(1);
                if self.played.0 == turn {
                    self.played.1 += 1;
                } else {
                    self.played = (turn, 1);
                }
                if !self.started && Some(turn) == self.current {
                    self.started = true;
                    let _ = events.send(RenderEvent::Started { reply: turn });
                }
                if self.pausing > 0 {
                    self.pausing -= 1;
                    if self.pausing == 0 {
                        self.paused = true;
                    }
                    s * self.gain * (self.pausing as f32 / PAUSE_FADE as f32)
                } else {
                    s * self.gain
                }
            }
            None => {
                if self.pausing > 0 {
                    self.pausing = 0;
                    self.paused = true;
                }
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
        // A paused reply is silent: Grace isn't talking while she lets someone else speak.
        let busy = self.audible_left > 0 && !self.paused;
        // Her voice is still in the room for the device buffer + speaker delay (Bluetooth
        // ~150 ms) after the last sample leaves the queue.
        if busy {
            self.last_busy = now;
        }
        talking.store(busy || now - self.last_busy < 0.5, Ordering::Relaxed);
        // The device buffer is kept full of silence, so "drained" means: the reply is complete,
        // its last audible sample was written, and the device buffer ahead of it has played.
        if busy || self.paused || !self.ended {
            self.drain_at = None;
        } else if let Some(turn) = self.current
            && self.started
        {
            match self.drain_at {
                None => self.drain_at = Some(now + device_queued as f64 / RATE as f64),
                Some(at) if now >= at => {
                    let _ = events.send(RenderEvent::Drained {
                        reply: turn,
                        time: now,
                    });
                    self.current = None;
                    self.drain_at = None;
                }
                Some(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(p: &mut Playback, n: usize) -> Vec<f32> {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        (0..n).map(|_| p.next_sample(&tx)).collect()
    }

    fn max_step(x: &[f32]) -> f32 {
        x.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn pause_and_resume_do_not_click() {
        let mut p = Playback::new();
        p.apply(RenderCmd::Enqueue {
            reply: 1,
            samples: vec![0.5; RATE as usize],
        });
        let mut out = play(&mut p, 100);
        p.apply(RenderCmd::Pause(true));
        out.extend(play(&mut p, PAUSE_FADE + 100));
        assert!(p.paused, "pause takes hold after the fade");
        assert_eq!(*out.last().unwrap(), 0.0);
        p.apply(RenderCmd::Pause(false));
        out.extend(play(&mut p, 2000));
        // A hard cut would be a 0.5 step; the fades keep every step tiny.
        assert!(max_step(&out) < 0.01, "max step {}", max_step(&out));
        assert!(
            (out.last().unwrap() - 0.5).abs() < 0.01,
            "back to full volume"
        );
    }

    #[test]
    fn pause_keeps_the_rest_of_the_reply() {
        let mut p = Playback::new();
        p.apply(RenderCmd::Enqueue {
            reply: 1,
            samples: vec![0.5; 1000],
        });
        p.apply(RenderCmd::Pause(true));
        play(&mut p, 5000);
        assert_eq!(p.queue.len(), 1000 - PAUSE_FADE);
    }

    #[test]
    fn cancel_fades_the_current_reply_and_keeps_the_next() {
        let mut p = Playback::new();
        p.apply(RenderCmd::Enqueue {
            reply: 1,
            samples: vec![0.5; RATE as usize],
        });
        let mut out = play(&mut p, 100);
        // Releasing a prepared reply sends its audio right after the cancel.
        p.apply(RenderCmd::Cancel);
        p.apply(RenderCmd::Enqueue {
            reply: 2,
            samples: vec![0.5; 1000],
        });
        assert_eq!(p.queue.len(), CANCEL_FADE + 1000);
        out.extend(play(&mut p, CANCEL_FADE + 10));
        assert!(
            out[100..100 + CANCEL_FADE].windows(2).all(|w| w[1] <= w[0]),
            "fades out"
        );
        assert_eq!(p.played, (2, 10));
        assert_eq!(p.current, Some(2));
    }

    #[test]
    fn cancelling_a_paused_reply_drops_it_silently() {
        let mut p = Playback::new();
        p.apply(RenderCmd::Enqueue {
            reply: 1,
            samples: vec![0.5; 1000],
        });
        play(&mut p, 100);
        p.apply(RenderCmd::Pause(true));
        play(&mut p, PAUSE_FADE + 50);
        assert_eq!(p.played, (1, 100 + PAUSE_FADE));
        p.apply(RenderCmd::Cancel);
        assert!(p.queue.is_empty());
        assert!(!p.paused);
        assert_eq!(p.audible_left, 0);
    }

    #[test]
    fn played_packs_reply_and_samples() {
        let played = Played::default();
        played.set((7, 123_456));
        assert_eq!(played.get(), (7, 123_456));
        assert_eq!(played.of(7), 123_456);
        assert_eq!(played.of(8), 0);
    }
}
