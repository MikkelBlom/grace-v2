//! The conductor: owns the turn machine and executes its commands.
//!
//! Speech edges come from the frontend; the conductor runs Smart Turn and STT when the user
//! pauses (in parallel: speculative transcription), prepares a reply while the turn end is still
//! uncertain (audio held until the commit), streams the LLM reply sentence by sentence into TTS
//! and the speakers, and cancels all of it at once when the user barges in.
//! Every stage is a trace span under its turn (ADR 0003).
//!
//! The conversation history and memory only ever hold what Mikkel heard: a reply is written
//! down when it is final (played to the end, or interrupted: only the part that played), and a
//! reply retracted because he hadn't finished (ADR 0018) is never written down at all.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::Duration,
};

use grace_core::turn::{Command, Config as TurnConfig, Input, TurnMachine};
use grace_llm::{
    client::{ChatRequest, LlmClient},
    sentences::SentenceSplitter,
};
use grace_vad::smart_turn::SmartTurn;
use serde_json::json;
use tokio::sync::{mpsc, watch};
use tracing::{Span, info, info_span, warn};

use crate::{
    audio_io::{Played, RATE, RenderCmd, RenderEvent},
    clock,
    frontend::{FrontEvent, Shared},
    speech::{Stt, Tts, TtsEvent, resample},
};

pub struct Services {
    pub llm: LlmClient,
    pub llm_model: String,
    pub stt: Arc<Stt>,
    pub tts: Arc<Tts>,
    pub smart_turn: Arc<Mutex<SmartTurn>>,
    pub shared: Arc<Mutex<Shared>>,
    pub render: std::sync::mpsc::Sender<RenderCmd>,
    /// How much of each reply the speakers have played.
    pub played: Played,
    pub system_prompt: PathBuf,
    pub memory: Arc<crate::memory::MemoryService>,
    /// Grace's tools and the decision checker (None: the tools server isn't running).
    pub tools: Option<Arc<crate::tools::Tools>>,
    /// Where each turn's STT input is saved for diagnosis (None = don't save).
    pub turn_audio_dir: Option<PathBuf>,
}

/// Results of work the conductor spawned, fed back into the machine.
enum Done {
    Score {
        turn: u64,
        epoch: u32,
        probability: f32,
    },
    Transcript {
        turn: u64,
        epoch: u32,
        text: String,
        /// Whisper was sure enough of the words to learn from them (avg_logprob ≥ -0.5).
        confident: bool,
    },
    /// A reply's generation ended: all its audio is queued (or it was cancelled).
    Generated { reply: u64, any_audio: bool },
    /// A reminder or watcher line was spoken unprompted (or `said` is None: nothing was due).
    Announced { said: Option<String> },
}

/// Quiet needed before Grace speaks up on her own: a due reminder soon, a watcher at a pause.
const QUIET_NOW_S: f64 = 1.0;
const QUIET_PAUSE_S: f64 = 4.0;

struct Turn {
    span: Span,
    start: f64,
    stop: f64,
    epoch: u32,
    play: Option<Span>,
}

/// Whether a reply's audio may play.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gate {
    /// Prepared before the commit: generate the first sentence, keep the audio.
    Held,
    Released,
    Cancelled,
}

type History = Vec<(&'static str, String)>;

struct Reply {
    /// The render id (a turn can have several replies: a retracted one and its replacement).
    id: u64,
    turn: u64,
    /// What Mikkel said: the user message this reply answers.
    text: String,
    /// The conversation the request was built from (without `text`): a prepared reply is only
    /// released if the conversation is still the same at the commit.
    context: History,
    gate: watch::Sender<Gate>,
    /// Sentences handed to the speakers, with their length in samples.
    sentences: Arc<Mutex<Vec<(String, usize)>>>,
    learn: bool,
    /// The user message taken out of the history while its turn is reopened.
    stash: Option<(&'static str, String)>,
    paused: bool,
    /// Generation ended without any audio: nothing will drain.
    silent: bool,
    /// Tool calls this reply made, as `tools::tool_note`s for the history.
    tool_notes: Arc<Mutex<Vec<String>>>,
    span: Span,
}

impl Reply {
    /// The sentences that `samples` of audio covered; a sentence cut short keeps its share of
    /// words and ends in a dash, so the model knows it was cut off.
    fn heard(&self, samples: usize) -> String {
        heard_text(&self.sentences.lock().expect("sentences poisoned"), samples)
    }
}

fn heard_text(sentences: &[(String, usize)], samples: usize) -> String {
    let (mut left, mut out) = (samples, Vec::new());
    for (sentence, len) in sentences {
        if left >= *len {
            out.push(sentence.clone());
            left -= len;
            continue;
        }
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let k = words.len() * left / (*len).max(1);
        if k > 0 {
            out.push(format!("{} —", words[..k].join(" ")));
        }
        break;
    }
    out.join(" ")
}

struct Conductor {
    svc: Arc<Services>,
    done_tx: mpsc::UnboundedSender<Done>,
    turns: HashMap<u64, Turn>,
    history: History,
    /// The reply that is playing, or paused by speech over it.
    current: Option<Reply>,
    /// A reply generated ahead of the commit, audio held.
    prepared: Option<Reply>,
    next_reply: u64,
    /// Turns whose transcript Whisper was unsure of: answered, but not learned from.
    unsure: HashSet<u64>,
    last_speech_start: f64,
    /// The open turn whose audio the rings keep whole, however long it runs.
    pinned: Option<u64>,
    speaking: bool,
    last_speech_stop: f64,
    /// An unprompted line (reminder, watcher) is being fetched or spoken.
    announcing: bool,
}

pub async fn run(
    svc: Services,
    mut front: mpsc::UnboundedReceiver<FrontEvent>,
    mut render_events: mpsc::UnboundedReceiver<RenderEvent>,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let t0 = clock::now();
    let mut machine = TurnMachine::new(TurnConfig::default());
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<Done>();
    let mut c = Conductor {
        svc: Arc::new(svc),
        done_tx,
        turns: HashMap::new(),
        history: Vec::new(),
        current: None,
        prepared: None,
        next_reply: 1,
        unsure: HashSet::new(),
        last_speech_start: t0,
        pinned: None,
        speaking: false,
        last_speech_stop: t0,
        announcing: false,
    };
    let mut tick = tokio::time::interval(Duration::from_millis(30));
    let mut events_tick = tokio::time::interval(Duration::from_secs(1));
    let rel = |t: f64| Duration::from_secs_f64((t - t0).max(0.0));

    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        let (input, now) = tokio::select! {
            Some(ev) = front.recv() => match ev {
                FrontEvent::SpeechStarted { time } => { c.last_speech_start = time; c.speaking = true; (Some(Input::SpeechStarted), time) }
                FrontEvent::SpeechStopped { time } => {
                    c.speaking = false;
                    c.last_speech_stop = time;
                    if let Some(t) = c.turns.values_mut().max_by(|a, b| a.start.total_cmp(&b.start)) { t.stop = time; }
                    (Some(Input::SpeechStopped), time)
                }
                FrontEvent::Delay { ms, confidence } => {
                    info!(delay_ms = ms, confidence = confidence as f64, "echo delay updated");
                    (None, clock::now())
                }
            },
            Some(ev) = render_events.recv() => match ev {
                RenderEvent::Started { reply } => (c.started(reply), clock::now()),
                RenderEvent::Drained { reply, time } => (c.drained(reply), time),
            },
            Some(d) = done_rx.recv() => match d {
                Done::Score { turn, epoch, probability } => {
                    let current = c.turns.get(&turn).is_some_and(|t| t.epoch == epoch);
                    (current.then_some(Input::TurnEndScore { turn, probability }), clock::now())
                }
                Done::Transcript { turn, epoch, text, confident } => {
                    let current = c.turns.get(&turn).is_some_and(|t| t.epoch == epoch);
                    if current && !confident { c.unsure.insert(turn); } else if current { c.unsure.remove(&turn); }
                    if current && let Some(tools) = &c.svc.tools && !text.trim().is_empty() {
                        tools.prefetch(&text);
                    }
                    (current.then_some(Input::Transcript { turn, text }), clock::now())
                }
                Done::Generated { reply, any_audio } => (c.generated(reply, any_audio), clock::now()),
                Done::Announced { said } => {
                    c.announcing = false;
                    if let Some(said) = said {
                        // So "udsæt den" / "hvad sagde du" has something to refer to.
                        c.history.push(("assistant", said));
                    }
                    (None, clock::now())
                }
            },
            _ = events_tick.tick() => {
                c.maybe_announce(clock::now());
                (None, clock::now())
            }
            _ = tick.tick() => (Some(Input::Tick), clock::now()),
        };
        let mut inputs: VecDeque<Input> = input.into_iter().collect();
        while let Some(input) = inputs.pop_front() {
            for cmd in machine.step(input, rel(now)) {
                inputs.extend(c.execute(cmd, now));
            }
        }
    }
    Ok(())
}

impl Conductor {
    /// Speaks a due reminder or a fired watcher when nobody is talking and no reply is under
    /// way. The line goes straight to TTS (it is already a sentence), so it costs no LLM time.
    fn maybe_announce(&mut self, now: f64) {
        let Some(tools) = self.svc.tools.clone() else { return };
        let quiet = now - self.last_speech_stop.max(self.last_speech_start);
        if self.announcing || self.speaking || self.current.is_some() || self.prepared.is_some()
            || quiet < QUIET_NOW_S
        {
            return;
        }
        self.announcing = true;
        let id = self.next_reply;
        self.next_reply += 1;
        let (svc, done) = (self.svc.clone(), self.done_tx.clone());
        tokio::spawn(async move {
            let said = announce(&svc, &tools, id, quiet).await;
            let _ = done.send(Done::Announced { said });
        });
    }

    fn render(&self, cmd: RenderCmd) {
        let _ = self.svc.render.send(cmd);
    }

    fn unpin(&mut self, id: u64) {
        if self.pinned == Some(id) {
            self.pinned = None;
            self.svc
                .shared
                .lock()
                .expect("shared poisoned")
                .keep_from(None);
        }
    }

    fn pin(&mut self, id: u64, start: f64) {
        // Pause work reads from 0.5 s before the start; keep a little more.
        self.svc
            .shared
            .lock()
            .expect("shared poisoned")
            .keep_from(Some(start - 1.0));
        self.pinned = Some(id);
    }

    fn span(&self, turn: u64) -> Span {
        self.turns
            .get(&turn)
            .map(|t| t.span.clone())
            .unwrap_or_else(Span::none)
    }

    fn started(&mut self, reply: u64) -> Option<Input> {
        let turn = self.current.as_ref().filter(|r| r.id == reply)?.turn;
        if let Some(t) = self.turns.get_mut(&turn) {
            t.play = Some(info_span!(parent: &t.span, "stage", stage = "tts.play"));
        }
        Some(Input::ReplyAudible { turn })
    }

    fn drained(&mut self, reply: u64) -> Option<Input> {
        let r = self.current.take_if(|r| r.id == reply)?;
        let turn = r.turn;
        if let Some(t) = self.turns.get_mut(&turn) {
            t.play = None;
        }
        let all = r.heard(usize::MAX);
        self.finish(r, all, false);
        Some(Input::ReplyFinished { turn })
    }

    fn generated(&mut self, reply: u64, any_audio: bool) -> Option<Input> {
        if any_audio {
            return None;
        }
        let r = self.current.as_mut().filter(|r| r.id == reply)?;
        if *r.gate.borrow() != Gate::Released {
            return None;
        }
        // Nothing will ever drain: the reply is over now (or when it is resumed).
        r.silent = true;
        if r.paused {
            return None;
        }
        let r = self.current.take()?;
        let turn = r.turn;
        self.finish(r, String::new(), false);
        Some(Input::ReplyFinished { turn })
    }

    /// Starts generating a reply to `text` after `context`; `gate` says whether it may play.
    fn spawn_reply(&mut self, turn: u64, text: String, context: History, gate: Gate) -> Reply {
        let id = self.next_reply;
        self.next_reply += 1;
        let (tx, rx) = watch::channel(gate);
        let sentences = Arc::new(Mutex::new(Vec::new()));
        let tool_notes = Arc::new(Mutex::new(Vec::new()));
        let span = self.span(turn);
        let mut messages = context.clone();
        messages.push(("user", text.clone()));
        tokio::spawn(speak_reply(
            self.svc.clone(),
            self.done_tx.clone(),
            id,
            messages,
            span.clone(),
            rx,
            ReplyOut {
                sentences: sentences.clone(),
                tool_notes: tool_notes.clone(),
            },
        ));
        Reply {
            id,
            turn,
            text,
            context,
            gate: tx,
            sentences,
            learn: true,
            stash: None,
            paused: false,
            silent: false,
            tool_notes,
            span,
        }
    }

    fn discard_prepared(&mut self, turn: Option<u64>) {
        if let Some(p) = self.prepared.take_if(|p| turn.is_none_or(|t| p.turn == t)) {
            p.gate.send_replace(Gate::Cancelled);
            info!(parent: &p.span, "prepared reply discarded");
        }
    }

    /// The reply is over: the conversation and memory get what Mikkel heard of it.
    fn finish(&mut self, r: Reply, heard: String, cut: bool) {
        r.gate.send_replace(Gate::Cancelled); // stops generation if it was still running
        self.unpin(r.turn);
        info!(parent: &r.span, text = %heard, cut, "reply heard");
        // What a tool found or did stays in the conversation, even if he cut the reply short.
        for note in r.tool_notes.lock().expect("notes poisoned").drain(..) {
            self.history.push(("tool", note));
        }
        if !heard.is_empty() {
            self.history.push(("assistant", heard.clone()));
        }
        prune_history(&mut self.history);
        println!("grace: {heard}{}", if cut { " (interrupted)" } else { "" });
        let memory = &self.svc.memory;
        let user_turn = memory.add_turn("user", &r.text).unwrap_or_else(|e| {
            warn!(error = %e, "memory: storing user turn failed");
            0
        });
        if !heard.is_empty()
            && let Err(e) = memory.add_turn("assistant", &heard)
        {
            warn!(error = %e, "memory: storing reply failed");
        }
        // Learn from what Mikkel said even when he interrupted the reply: in live run 3 the turns
        // with the most facts were the interrupted ones, and they were never learned. Runs in the
        // background on the LLM's second slot, never on the voice path.
        if r.learn {
            let reply = if heard.is_empty() {
                "(Grace was interrupted before she could answer.)".to_owned()
            } else {
                heard
            };
            tokio::spawn(memory.clone().learn(user_turn, r.text, reply));
        } else {
            info!("memory: transcript too uncertain to learn from");
        }
    }

    /// Executes one command; returns inputs it caused (fed to the machine right away).
    fn execute(&mut self, cmd: Command, now: f64) -> Vec<Input> {
        match cmd {
            Command::BeginTurn(id) => {
                let span = info_span!("turn", turn = format!("t{id}"));
                self.turns.insert(
                    id,
                    Turn {
                        span,
                        start: self.last_speech_start,
                        stop: now,
                        epoch: 0,
                        play: None,
                    },
                );
                // Keep the map small: only recent turns matter.
                self.turns.retain(|k, _| *k + 8 > id);
                self.pin(id, self.last_speech_start);
            }
            Command::ReopenTurn(id) => {
                let Some(t) = self.turns.get_mut(&id) else {
                    return Vec::new();
                };
                t.epoch += 1;
                info!(parent: &t.span, "turn reopened: he was not done");
                let start = t.start;
                self.pin(id, start);
                // His first part leaves the conversation until we know whether it stands alone.
                if let Some(r) = self.current.as_mut().filter(|r| r.turn == id)
                    && r.stash.is_none()
                    && self
                        .history
                        .last()
                        .is_some_and(|(role, text)| *role == "user" && *text == r.text)
                {
                    r.stash = self.history.pop();
                }
                println!("       (listening on)");
            }
            Command::EvaluatePause(id) => {
                let Some(turn) = self.turns.get_mut(&id) else {
                    return Vec::new();
                };
                turn.epoch += 1;
                let now_c = clock::now();
                let us = |s: f64| (s.max(0.0) * 1e6) as u64;
                drop(
                    info_span!(parent: &turn.span, "stage", stage = "vad.speech", model = "silero-v6.2.3",
                    backdate_us = us(now_c - turn.start), trim_us = us(now_c - turn.stop)),
                );
                spawn_pause_work(
                    self.svc.clone(),
                    self.done_tx.clone(),
                    id,
                    turn.epoch,
                    turn.start,
                    turn.stop,
                    turn.span.clone(),
                );
            }
            Command::DiscardTranscript(id) => {
                if let Some(t) = self.turns.get_mut(&id) {
                    t.epoch += 1;
                }
                self.discard_prepared(Some(id));
            }
            Command::Prepare { turn, text } => {
                self.discard_prepared(None);
                let mut context = self.history.clone();
                // A reply this turn interrupted will be in the conversation, cut where it was
                // paused, by the time this turn is answered.
                if let Some(r) = self.current.as_ref().filter(|r| r.turn != turn && r.paused) {
                    let heard = r.heard(self.svc.played.of(r.id));
                    if !heard.is_empty() {
                        context.push(("assistant", heard));
                    }
                    prune_history(&mut context);
                }
                let reply = self.spawn_reply(turn, text, context, Gate::Held);
                info!(parent: &reply.span, "reply prepared");
                self.prepared = Some(reply);
            }
            Command::Respond { turn, text } => {
                // Stay pinned until the reply is done: he may go on talking and reopen the turn,
                // and its re-transcription needs the audio from the start. Live run 5 lost the
                // first 4 s of a 35 s turn when this unpinned here and the ring evicted them.
                println!("you:   {text}");
                if let Some(old) = self.current.take() {
                    // The machine retracts or cancels first; this is only a safety net.
                    old.gate.send_replace(Gate::Cancelled);
                    self.render(RenderCmd::Cancel);
                }
                // Live run 3 stored "Mikkel's files are labeled OGEEEEEEE" from a spelled-out
                // word Whisper scored -0.69: memory only learns from confident transcripts.
                let learn = !self.unsure.remove(&turn);
                let mut reply = match self.prepared.take() {
                    Some(p) if p.turn == turn && p.text == text && p.context == self.history => {
                        p.gate.send_replace(Gate::Released);
                        info!(parent: &p.span, "prepared reply released");
                        p
                    }
                    other => {
                        if let Some(p) = other {
                            p.gate.send_replace(Gate::Cancelled);
                        }
                        let context = self.history.clone();
                        self.spawn_reply(turn, text.clone(), context, Gate::Released)
                    }
                };
                reply.learn = learn;
                self.history.push(("user", text));
                self.current = Some(reply);
            }
            Command::DropTurn(id) => {
                self.unpin(id);
                self.discard_prepared(Some(id));
                if let Some(t) = self.turns.get(&id) {
                    info!(parent: &t.span, "turn dropped: no words");
                }
                self.turns.remove(&id);
            }
            Command::DuckReply => self.render(RenderCmd::Duck(true)),
            Command::UnduckReply => self.render(RenderCmd::Duck(false)),
            Command::PauseReply(id) => {
                self.render(RenderCmd::Pause(true));
                self.render(RenderCmd::Duck(false));
                if let Some(r) = self.current.as_mut().filter(|r| r.turn == id) {
                    r.paused = true;
                    info!(parent: &r.span, "reply paused");
                }
            }
            Command::ResumeReply(id) => {
                self.render(RenderCmd::Pause(false));
                self.unpin(id);
                self.discard_prepared(Some(id));
                let Some(r) = self.current.as_mut().filter(|r| r.turn == id) else {
                    return Vec::new();
                };
                r.paused = false;
                if let Some(entry) = r.stash.take() {
                    self.history.push(entry);
                }
                info!(parent: &r.span, "reply resumed: nothing new was said");
                println!("       (carries on)");
                if r.silent {
                    let r = self.current.take().expect("checked above");
                    self.finish(r, String::new(), false);
                    return vec![Input::ReplyFinished { turn: id }];
                }
            }
            Command::CancelReply(id) => {
                self.render(RenderCmd::Cancel);
                self.render(RenderCmd::Duck(false));
                if let Some(r) = self.current.take_if(|r| r.turn == id) {
                    let heard = r.heard(self.svc.played.of(r.id));
                    info!(parent: &r.span, "reply interrupted");
                    self.finish(r, heard, true);
                }
            }
            Command::RetractReply(id) => {
                self.render(RenderCmd::Cancel);
                self.render(RenderCmd::Duck(false));
                // Never said: no history, no memory; the whole turn is answered instead.
                if let Some(r) = self.current.take_if(|r| r.turn == id) {
                    r.gate.send_replace(Gate::Cancelled);
                    let heard = r.heard(self.svc.played.of(r.id));
                    info!(parent: &r.span, heard = %heard, "reply retracted: he was not done");
                    println!("       (retracted; answering the whole turn)");
                }
            }
        }
        Vec::new()
    }
}

/// Below this mean token log-probability a transcript is answered but not learned from.
const LEARN_LOGPROB: f64 = -0.5;

/// Smart Turn and STT in parallel on the paused turn (speculative transcription).
fn spawn_pause_work(
    svc: Arc<Services>,
    done: mpsc::UnboundedSender<Done>,
    turn: u64,
    epoch: u32,
    start: f64,
    stop: f64,
    span: Span,
) {
    // Smart Turn reads the full AEC3 chain (ADR 0013); STT reads AEC3's linear output, without
    // residual-echo and noise suppression, whose artifacts cost 5-7 WER points in grace-sim
    // (ADR 0014).
    let (aec, stt_audio, raw, full, source) = {
        let shared = svc.shared.lock().expect("shared poisoned");
        (
            shared.aec16.read_span(start - 0.5, stop + 0.2),
            shared.lin16.read_span(start - 0.3, stop + 0.15),
            shared.raw16.read_span(start - 0.3, stop + 0.15),
            shared.aec16.read_span(start - 0.3, stop + 0.15),
            "lin",
        )
    };
    // Keep what STT heard (and the raw mic and full chain) next to the trace: a misheard turn can
    // be listened to and re-transcribed offline. Local only; traces/ is gitignored.
    if let Some(dir) = &svc.turn_audio_dir {
        let (l, r) = (stt_audio.clone(), raw);
        let base = dir.join(format!("t{turn}-{epoch}"));
        tokio::task::spawn_blocking(move || {
            for (suffix, pcm) in [("lin", l), ("raw", r), ("aec", full)] {
                let _ = crate::speech::write_wav(
                    &base.with_extension(format!("{suffix}.wav")),
                    &pcm,
                    16_000,
                );
            }
        });
    }
    let (svc2, done2, span2) = (svc.clone(), done.clone(), span.clone());
    tokio::task::spawn_blocking(move || {
        let stage = info_span!(parent: &span2, "stage", stage = "turn.detect", model = "smart-turn-v3.2-da1");
        let p = svc2
            .smart_turn
            .lock()
            .expect("smart turn poisoned")
            .predict(&aec);
        drop(stage);
        match p {
            Ok(probability) => {
                info!(parent: &span2, probability = probability as f64, "turn end score");
                let _ = done2.send(Done::Score {
                    turn,
                    epoch,
                    probability,
                });
            }
            Err(e) => warn!(parent: &span2, error = %e, "smart turn failed"),
        }
    });
    tokio::spawn(async move {
        let stage =
            info_span!(parent: &span, "stage", stage = "stt", model = "edda-v0.2", source = source);
        let result = svc.stt.transcribe(&stt_audio).await;
        drop(stage);
        match result {
            Ok(t) => {
                let hallucinated = t.looks_hallucinated();
                info!(parent: &span, text = %t.text, check = %t.check_text, server_ms = t.ms, avg_logprob = t.avg_logprob,
                    no_speech_prob = t.no_speech_prob, hallucinated, "transcript");
                // A dropped transcript ends the turn as "no words" (DropTurn), nothing is answered.
                let text = if hallucinated { String::new() } else { t.text };
                let confident = t.avg_logprob >= LEARN_LOGPROB;
                let _ = done.send(Done::Transcript {
                    turn,
                    epoch,
                    text,
                    confident,
                });
            }
            Err(e) => {
                warn!(parent: &span, error = %e, "stt failed");
                let _ = done.send(Done::Transcript {
                    turn,
                    epoch,
                    text: String::new(),
                    confident: false,
                });
            }
        }
    });
}

/// Keeps the conversation within the context budget without invalidating the LLM's prompt cache
/// every turn: history only grows until it passes ~5000 tokens, then the oldest half goes at once.
/// (Sliding a fixed window by one message per turn changed the prefix every turn and pushed
/// time-to-first-token from ~60 ms to ~500 ms in the first live run.)
fn prune_history(history: &mut Vec<(&'static str, String)>) {
    let tokens = |h: &[(&str, String)]| h.iter().map(|(_, t)| t.len() / 3 + 4).sum::<usize>();
    if tokens(history) <= 5000 {
        return;
    }
    while tokens(history) > 2500 && history.len() > 2 {
        history.remove(0);
    }
    // Start on a user turn so roles keep alternating.
    while history.first().is_some_and(|(r, _)| *r != "user") {
        history.remove(0);
    }
}

/// The whole system message: the prompt file, the date, and this session's memory block.
/// Shared with the memory benchmark, so its probes are asked exactly as Grace is asked.
pub fn system_message(
    path: &std::path::Path,
    now: chrono::DateTime<chrono::Local>,
    memory_block: &str,
) -> String {
    // Read every turn: editing the file changes Grace's behaviour immediately (ADR 0002).
    let base = std::fs::read_to_string(path)
        .unwrap_or_else(|_| "You are Grace, a helpful voice assistant.".into());
    // Hour granularity keeps the cached prompt prefix valid for an hour at a time. The memory
    // block comes first: Gemma's template puts the tool declarations right after the system
    // text, and with memory last she asked for facts it already listed (chat bench 02 with tools).
    format!(
        "{memory_block}\n\n{base}\n\nToday is {}, around {}:00 local time (Denmark).",
        now.format("%A %-d %B %Y"),
        now.format("%H")
    )
}

/// What a reply leaves behind for the conductor: the sentences that went to the speakers and
/// the tool calls it made.
struct ReplyOut {
    sentences: Arc<Mutex<Vec<(String, usize)>>>,
    tool_notes: Arc<Mutex<Vec<String>>>,
}

/// Where a reply's words go: sentences to TTS, and the gate that says whether it may act.
struct ReplySink<'a> {
    sent_tx: &'a mpsc::UnboundedSender<String>,
    splitter: &'a mut SentenceSplitter,
    full: &'a mut String,
    gate: watch::Receiver<Gate>,
    ttft: &'a mut Option<Span>,
    tool_notes: &'a Mutex<Vec<String>>,
}

impl crate::tools::Sink for ReplySink<'_> {
    fn text(&mut self, text: &str) {
        self.ttft.take();
        self.full.push_str(text);
        for piece in self.splitter.push(text) {
            let _ = self.sent_tx.send(piece);
        }
    }

    fn filler(&mut self, line: &str) {
        self.ttft.take();
        self.full.push_str(line);
        self.full.push(' ');
        let _ = self.sent_tx.send(line.to_owned());
    }

    fn tool_done(&mut self, name: &str, args: &serde_json::Value, out: &serde_json::Value) {
        self.tool_notes
            .lock()
            .expect("notes poisoned")
            .push(crate::tools::tool_note(name, args, out));
    }

    async fn cancelled(&mut self) {
        let _ = self.gate.wait_for(|g| *g == Gate::Cancelled).await;
    }

    fn is_cancelled(&self) -> bool {
        *self.gate.borrow() == Gate::Cancelled
    }

    async fn ready_to_act(&mut self) -> bool {
        let _ = self.gate.wait_for(|g| *g != Gate::Held).await;
        *self.gate.borrow() == Gate::Released
    }

    fn silent(&self) -> bool {
        self.full.trim().is_empty()
    }
}

/// Streams the LLM reply into TTS and the speakers, one sentence at a time. A held reply
/// (prepared before the commit) synthesizes only its first sentence and keeps the audio until
/// it is released.
async fn announce(svc: &Services, tools: &crate::tools::Tools, reply: u64, quiet: f64) -> Option<String> {
    let events = match tools.events(&[]).await {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "events poll failed");
            return None;
        }
    };
    let ev = events.into_iter().find(|e| e["priority"] == "now" || quiet >= QUIET_PAUSE_S)?;
    let say = ev["say"].as_str()?.to_owned();
    info!(source = %ev["source"], say = %say, "announcing");
    let (id, mut audio) = svc.tts.speak(&crate::spoken::numbers_to_words(&say)).await.ok()?;
    let _ = id;
    while let Some(ev) = audio.recv().await {
        match ev {
            TtsEvent::Audio { rate, pcm } => {
                let _ = svc.render.send(RenderCmd::Enqueue { reply, samples: resample(&pcm, rate, RATE) });
            }
            TtsEvent::Done { .. } => break,
        }
    }
    let _ = svc.render.send(RenderCmd::EndOfTurn { reply });
    // Acked once it is queued to play: at-least-once until then (a crash re-announces it).
    if let Some(n) = ev["id"].as_i64()
        && let Err(e) = tools.events(&[n]).await
    {
        warn!(error = %e, "events ack failed");
    }
    Some(say)
}

async fn speak_reply(
    svc: Arc<Services>,
    done: mpsc::UnboundedSender<Done>,
    reply: u64,
    history: History,
    span: Span,
    gate: watch::Receiver<Gate>,
    out: ReplyOut,
) {
    let ReplyOut {
        sentences,
        tool_notes,
    } = out;
    let cancelled = |g: &watch::Receiver<Gate>| *g.borrow() == Gate::Cancelled;
    // The system prompt (live file + this session's memory block) is fixed for the session, so
    // llama-server's prompt cache stays valid across turns; only a new style rule changes it.
    let system = system_message(
        &svc.system_prompt,
        chrono::Local::now(),
        &svc.memory.prompt_block(),
    );
    let mut messages = vec![json!({ "role": "system", "content": system })];
    // Full history (bounded by `prune_history`): a stable prefix keeps llama-server's cache valid.
    let (last, earlier) = history
        .split_last()
        .expect("history ends with the user's turn");
    messages.extend(crate::tools::to_messages(earlier));
    // Recall goes on the current user message only: the cached prefix before it is untouched.
    // The checker's "needs a lookup?" runs next to recall and is dropped if it is late: with
    // its hint Gemma searched on 17/17 lookup requests instead of 14/17 (ADR 0019).
    let recall = async {
        let _stage =
            info_span!(parent: &span, "stage", stage = "memory.recall", model = "harrier-270m");
        match svc.memory.recall(&last.1).await {
            Ok(facts) => facts,
            Err(e) => {
                warn!(parent: &span, error = %e, "memory recall failed");
                Vec::new()
            }
        }
    };
    // The hint may finish a little after recall, never much: live run 9 waited 288 ms p50 for it
    // while recall took 121 ms.
    let hint = tokio::spawn({
        let (svc, said, span) = (svc.clone(), last.1.clone(), span.clone());
        async move {
            match svc.tools.as_ref() {
                Some(t) => t.hint(&said, &span).await,
                None => None,
            }
        }
    });
    let recalled = recall.await;
    let needs_lookup = tokio::time::timeout(std::time::Duration::from_millis(80), hint)
        .await
        .ok()
        .and_then(Result::ok)
        .flatten();
    if !recalled.is_empty() {
        let texts: Vec<&str> = recalled.iter().map(|f| f.text.as_str()).collect();
        info!(parent: &span, recalled = ?texts, "memory recalled");
    }
    // The style rules are restated next to his words, for this turn only: in the system block
    // alone, a "fewer questions" rule held in-session but not in the next one (ADR 0017).
    let rules = svc.memory.style_rules().unwrap_or_default();
    let mut content =
        crate::style::with_reminder(&crate::memory::with_recall(&last.1, &recalled), &rules);
    if needs_lookup.is_some_and(|r| r.hint()) {
        content = format!("{content}\n\n{}", crate::tools::HINT);
    }
    messages.push(json!({ "role": "user", "content": content }));
    let mut req = ChatRequest::new(messages);
    req.max_tokens = 300;

    // Sentences flow to a TTS worker as soon as they are complete.
    let (sent_tx, mut sent_rx) = mpsc::unbounded_channel::<String>();
    let tts_task = {
        let (svc, span, mut gate) = (svc.clone(), span.clone(), gate.clone());
        tokio::spawn(async move {
            let mut ttfa =
                Some(info_span!(parent: &span, "stage", stage = "tts.ttfa", model = "pocket-tts"));
            let mut any_audio = false;
            // Audio made before the release.
            let mut held: Vec<f32> = Vec::new();
            let flush = |held: &mut Vec<f32>, gate: &watch::Receiver<Gate>| {
                if *gate.borrow() == Gate::Released && !held.is_empty() {
                    let _ = svc.render.send(RenderCmd::Enqueue {
                        reply,
                        samples: std::mem::take(held),
                    });
                }
            };
            let mut first = true;
            'sentences: while let Some(sentence) = sent_rx.recv().await {
                // Before the commit only the first sentence is synthesized: Pocket TTS shares
                // the CPU with Smart Turn, and a discarded reply should cost little.
                if !first && *gate.borrow() == Gate::Held {
                    let _ = gate.wait_for(|g| *g != Gate::Held).await;
                    flush(&mut held, &gate);
                }
                first = false;
                if cancelled(&gate) {
                    break;
                }
                let Ok((id, mut audio)) = svc
                    .tts
                    .speak(&crate::spoken::numbers_to_words(&sentence))
                    .await
                else {
                    break;
                };
                sentences
                    .lock()
                    .expect("sentences poisoned")
                    .push((sentence.clone(), 0));
                loop {
                    let ev = tokio::select! {
                        ev = audio.recv() => ev,
                        changed = gate.changed() => {
                            if changed.is_err() || cancelled(&gate) {
                                svc.tts.cancel(id).await;
                                break 'sentences;
                            }
                            flush(&mut held, &gate); // released mid-sentence
                            continue;
                        }
                    };
                    match ev {
                        Some(TtsEvent::Audio { rate, pcm }) => {
                            ttfa.take();
                            any_audio = true;
                            let pcm = resample(&pcm, rate, RATE);
                            if let Some(last) =
                                sentences.lock().expect("sentences poisoned").last_mut()
                            {
                                last.1 += pcm.len();
                            }
                            held.extend(pcm);
                            flush(&mut held, &gate);
                        }
                        Some(TtsEvent::Done {
                            cancelled,
                            first_ms,
                        }) => {
                            info!(parent: &span, server_first_ms = first_ms.unwrap_or(-1.0), sentence = %sentence, "tts sentence");
                            if cancelled {
                                break 'sentences;
                            }
                            break;
                        }
                        None => break 'sentences,
                    }
                }
            }
            // A prepared reply that is complete waits here for its commit.
            if *gate.borrow() == Gate::Held {
                let _ = gate.wait_for(|g| *g != Gate::Held).await;
            }
            flush(&mut held, &gate);
            any_audio
        })
    };

    let ttft =
        info_span!(parent: &span, "stage", stage = "llm.ttft", model = svc.llm_model.as_str());
    let mut ttft = Some(ttft);
    let decode =
        info_span!(parent: &span, "stage", stage = "llm.decode", model = svc.llm_model.as_str());
    let mut splitter = SentenceSplitter::default();
    let mut full = String::new();
    let mut sink = ReplySink {
        sent_tx: &sent_tx,
        splitter: &mut splitter,
        full: &mut full,
        gate: gate.clone(),
        ttft: &mut ttft,
        tool_notes: &tool_notes,
    };
    crate::tools::reply(
        &svc.llm,
        svc.tools.as_deref(),
        req,
        reply,
        &last.1,
        &span,
        &mut sink,
    )
    .await;
    if !cancelled(&gate)
        && let Some(rest) = splitter.finish()
    {
        let _ = sent_tx.send(rest);
    }
    drop(decode);
    drop(ttft);
    drop(sent_tx);
    info!(parent: &span, text = %full, "reply");
    let any_audio = tts_task.await.unwrap_or_default();
    // All of this reply's audio is queued: once it has played, the reply is over.
    if *gate.borrow() == Gate::Released {
        let _ = svc.render.send(RenderCmd::EndOfTurn { reply });
    }
    let _ = done.send(Done::Generated { reply, any_audio });
}

#[cfg(test)]
mod tests {
    use super::heard_text;

    #[test]
    fn heard_text_keeps_only_what_played() {
        let s = vec![
            ("Sure, I can do that.".to_owned(), 1000),
            (
                "First open the settings page and then pick audio.".to_owned(),
                2000,
            ),
        ];
        assert_eq!(heard_text(&s, 0), "");
        assert_eq!(heard_text(&s, 100), "");
        assert_eq!(heard_text(&s, 999), "Sure, I can do —");
        assert_eq!(heard_text(&s, 1000), "Sure, I can do that.");
        // Half of the second sentence: half its words, marked as cut off.
        assert_eq!(
            heard_text(&s, 2000),
            "Sure, I can do that. First open the settings —"
        );
        assert_eq!(
            heard_text(&s, usize::MAX),
            "Sure, I can do that. First open the settings page and then pick audio."
        );
    }
}
