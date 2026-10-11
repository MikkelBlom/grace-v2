//! The conductor: owns the turn machine and executes its commands.
//!
//! Speech edges come from the frontend; the conductor runs Smart Turn and STT when the user
//! pauses (in parallel: speculative transcription), streams the LLM reply sentence by sentence
//! into TTS and the speakers, and cancels all of it at once when the user barges in.
//! Every stage is a trace span under its turn (ADR 0003).

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use futures_util::StreamExt;
use grace_core::turn::{Command, Config as TurnConfig, Input, TurnMachine};
use grace_llm::{
    client::{ChatRequest, Delta, LlmClient},
    sentences::SentenceSplitter,
};
use grace_vad::smart_turn::SmartTurn;
use serde_json::json;
use tokio::sync::{Notify, mpsc};
use tracing::{Span, info, info_span, warn};

use crate::{
    audio_io::{RATE, RenderCmd, RenderEvent},
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
    pub system_prompt: PathBuf,
    pub memory: Arc<crate::memory::MemoryService>,
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
    },
    /// The reply produced no audio (empty or failed); nothing will drain.
    Silent {
        turn: u64,
    },
    Spoken {
        text: String,
    },
}

struct Turn {
    span: Span,
    start: f64,
    stop: f64,
    epoch: u32,
    play: Option<Span>,
}

struct Reply {
    turn: u64,
    cancel: Arc<AtomicBool>,
    wake: Arc<Notify>,
}

pub async fn run(
    svc: Services,
    mut front: mpsc::UnboundedReceiver<FrontEvent>,
    mut render_events: mpsc::UnboundedReceiver<RenderEvent>,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let svc = Arc::new(svc);
    let t0 = clock::now();
    let mut machine = TurnMachine::new(TurnConfig::default());
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<Done>();
    let mut turns: HashMap<u64, Turn> = HashMap::new();
    let mut history: Vec<(&'static str, String)> = Vec::new();
    let mut reply: Option<Reply> = None;
    let mut last_speech_start = t0;
    let mut tick = tokio::time::interval(Duration::from_millis(30));
    let rel = |t: f64| Duration::from_secs_f64((t - t0).max(0.0));

    while !shutdown.load(Ordering::Relaxed) {
        let (input, now) = tokio::select! {
            Some(ev) = front.recv() => match ev {
                FrontEvent::SpeechStarted { time } => { last_speech_start = time; (Some(Input::SpeechStarted), time) }
                FrontEvent::SpeechStopped { time } => {
                    if let Some(t) = turns.values_mut().max_by(|a, b| a.start.total_cmp(&b.start)) { t.stop = time; }
                    (Some(Input::SpeechStopped), time)
                }
                FrontEvent::Delay { ms, confidence } => {
                    info!(delay_ms = ms, confidence = confidence as f64, "echo delay updated");
                    (None, clock::now())
                }
            },
            Some(ev) = render_events.recv() => match ev {
                RenderEvent::Started { turn } => {
                    if let Some(t) = turns.get_mut(&turn) {
                        t.play = Some(info_span!(parent: &t.span, "stage", stage = "tts.play"));
                    }
                    (None, clock::now())
                }
                RenderEvent::Drained { turn, time } => {
                    if let Some(t) = turns.get_mut(&turn) { t.play = None; }
                    (Some(Input::ReplyFinished { turn }), time)
                }
            },
            Some(d) = done_rx.recv() => match d {
                Done::Score { turn, epoch, probability } => {
                    let current = turns.get(&turn).is_some_and(|t| t.epoch == epoch);
                    (current.then_some(Input::TurnEndScore { turn, probability }), clock::now())
                }
                Done::Transcript { turn, epoch, text } => {
                    let current = turns.get(&turn).is_some_and(|t| t.epoch == epoch);
                    (current.then_some(Input::Transcript { turn, text }), clock::now())
                }
                Done::Silent { turn } => (Some(Input::ReplyFinished { turn }), clock::now()),
                Done::Spoken { text } => {
                    if !text.is_empty() { history.push(("assistant", text)); }
                    prune_history(&mut history);
                    (None, clock::now())
                }
            },
            _ = tick.tick() => (Some(Input::Tick), clock::now()),
        };
        let Some(input) = input else { continue };
        for cmd in machine.step(input, rel(now)) {
            match cmd {
                Command::BeginTurn(id) => {
                    let span = info_span!("turn", turn = format!("t{id}"));
                    turns.insert(
                        id,
                        Turn {
                            span,
                            start: last_speech_start,
                            stop: now,
                            epoch: 0,
                            play: None,
                        },
                    );
                    // Keep the map small: only recent turns matter.
                    turns.retain(|k, _| *k + 8 > id);
                }
                Command::EvaluatePause(id) => {
                    let Some(turn) = turns.get_mut(&id) else {
                        continue;
                    };
                    turn.epoch += 1;
                    let now_c = clock::now();
                    let us = |s: f64| (s.max(0.0) * 1e6) as u64;
                    drop(
                        info_span!(parent: &turn.span, "stage", stage = "vad.speech", model = "silero-v6.2.3",
                        backdate_us = us(now_c - turn.start), trim_us = us(now_c - turn.stop)),
                    );
                    spawn_pause_work(
                        svc.clone(),
                        done_tx.clone(),
                        id,
                        turn.epoch,
                        turn.start,
                        turn.stop,
                        turn.span.clone(),
                    );
                }
                Command::DiscardTranscript(id) => {
                    if let Some(t) = turns.get_mut(&id) {
                        t.epoch += 1;
                    }
                }
                Command::Respond { turn, text } => {
                    println!("you:   {text}");
                    let user_turn = svc.memory.add_turn("user", &text).unwrap_or_else(|e| {
                        warn!(error = %e, "memory: storing user turn failed");
                        0
                    });
                    history.push(("user", text));
                    let cancel = Arc::new(AtomicBool::new(false));
                    let wake = Arc::new(Notify::new());
                    let span = turns
                        .get(&turn)
                        .map(|t| t.span.clone())
                        .unwrap_or_else(Span::none);
                    tokio::spawn(speak_reply(
                        svc.clone(),
                        done_tx.clone(),
                        turn,
                        history.clone(),
                        user_turn,
                        span,
                        cancel.clone(),
                        wake.clone(),
                    ));
                    reply = Some(Reply { turn, cancel, wake });
                }
                Command::DropTurn(id) => {
                    if let Some(t) = turns.get(&id) {
                        info!(parent: &t.span, "turn dropped: no words");
                    }
                    turns.remove(&id);
                }
                Command::DuckReply => {
                    let _ = svc.render.send(RenderCmd::Duck(true));
                }
                Command::UnduckReply => {
                    let _ = svc.render.send(RenderCmd::Duck(false));
                }
                Command::CancelReply(id) => {
                    let _ = svc.render.send(RenderCmd::Cancel);
                    let _ = svc.render.send(RenderCmd::Duck(false));
                    if let Some(r) = reply.take_if(|r| r.turn == id) {
                        r.cancel.store(true, Ordering::Relaxed);
                        r.wake.notify_waiters();
                    }
                    if let Some(t) = turns.get(&id) {
                        info!(parent: &t.span, "reply interrupted");
                    }
                    println!("       (interrupted)");
                }
            }
        }
    }
    Ok(())
}

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
        let stage =
            info_span!(parent: &span2, "stage", stage = "turn.detect", model = "smart-turn-v3.2");
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
                let _ = done.send(Done::Transcript { turn, epoch, text });
            }
            Err(e) => {
                warn!(parent: &span, error = %e, "stt failed");
                let _ = done.send(Done::Transcript {
                    turn,
                    epoch,
                    text: String::new(),
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
    // Hour granularity keeps the cached prompt prefix valid for an hour at a time.
    format!(
        "{base}\n\nToday is {}, around {}:00 local time (Denmark).\n\n{memory_block}",
        now.format("%A %-d %B %Y"),
        now.format("%H")
    )
}

/// Streams the LLM reply into TTS and the speakers, one sentence at a time.
#[allow(clippy::too_many_arguments)] // a spawned task's inputs; a struct would only rename them
async fn speak_reply(
    svc: Arc<Services>,
    done: mpsc::UnboundedSender<Done>,
    turn: u64,
    history: Vec<(&'static str, String)>,
    user_turn: i64,
    span: Span,
    cancel: Arc<AtomicBool>,
    wake: Arc<Notify>,
) {
    // The system prompt (live file + this session's memory block) is fixed for the session, so
    // llama-server's prompt cache stays valid across turns.
    let system = system_message(
        &svc.system_prompt,
        chrono::Local::now(),
        svc.memory.session_block(),
    );
    let mut messages = vec![json!({ "role": "system", "content": system })];
    // Full history (bounded by `prune_history`): a stable prefix keeps llama-server's cache valid.
    let (last, earlier) = history
        .split_last()
        .expect("history ends with the user's turn");
    messages.extend(
        earlier
            .iter()
            .map(|(r, t)| json!({ "role": r, "content": t })),
    );
    // Recall goes on the current user message only: the cached prefix before it is untouched.
    let recall_span =
        info_span!(parent: &span, "stage", stage = "memory.recall", model = "harrier-270m");
    let recalled = match svc.memory.recall(&last.1).await {
        Ok(facts) => facts,
        Err(e) => {
            warn!(parent: &span, error = %e, "memory recall failed");
            Vec::new()
        }
    };
    drop(recall_span);
    if !recalled.is_empty() {
        let texts: Vec<&str> = recalled.iter().map(|f| f.text.as_str()).collect();
        info!(parent: &span, recalled = ?texts, "memory recalled");
    }
    messages
        .push(json!({ "role": "user", "content": crate::memory::with_recall(&last.1, &recalled) }));
    let mut req = ChatRequest::new(messages);
    req.max_tokens = 300;

    // Sentences flow to a TTS worker as soon as they are complete.
    let (sent_tx, mut sent_rx) = mpsc::unbounded_channel::<String>();
    let tts_task = {
        let (svc, span, cancel, wake) = (svc.clone(), span.clone(), cancel.clone(), wake.clone());
        tokio::spawn(async move {
            let mut ttfa =
                Some(info_span!(parent: &span, "stage", stage = "tts.ttfa", model = "pocket-tts"));
            let (mut spoken, mut any_audio) = (String::new(), false);
            while let Some(sentence) = sent_rx.recv().await {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let Ok((id, mut audio)) = svc.tts.speak(&sentence).await else {
                    break;
                };
                let mut completed = false;
                loop {
                    let ev =
                        tokio::select! { ev = audio.recv() => ev, _ = wake.notified() => None };
                    if cancel.load(Ordering::Relaxed) {
                        svc.tts.cancel(id).await;
                        break;
                    }
                    match ev {
                        Some(TtsEvent::Audio { rate, pcm }) => {
                            ttfa.take();
                            any_audio = true;
                            let _ = svc.render.send(RenderCmd::Enqueue {
                                turn,
                                samples: resample(&pcm, rate, RATE),
                            });
                        }
                        Some(TtsEvent::Done {
                            cancelled,
                            first_ms,
                        }) => {
                            info!(parent: &span, server_first_ms = first_ms.unwrap_or(-1.0), sentence = %sentence, "tts sentence");
                            completed = !cancelled;
                            break;
                        }
                        None => break,
                    }
                }
                if completed {
                    if !spoken.is_empty() {
                        spoken.push(' ');
                    }
                    spoken.push_str(&sentence);
                } else {
                    break;
                }
            }
            (spoken, any_audio)
        })
    };

    let ttft =
        info_span!(parent: &span, "stage", stage = "llm.ttft", model = svc.llm_model.as_str());
    let mut ttft = Some(ttft);
    let decode =
        info_span!(parent: &span, "stage", stage = "llm.decode", model = svc.llm_model.as_str());
    let mut splitter = SentenceSplitter::default();
    let mut full = String::new();
    match svc.llm.stream(&req).await {
        Ok(mut stream) => loop {
            let delta = tokio::select! { d = stream.next() => d, _ = wake.notified() => None };
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            match delta {
                Some(Ok(Delta::Content(text))) => {
                    ttft.take();
                    full.push_str(&text);
                    for piece in splitter.push(&text) {
                        let _ = sent_tx.send(piece);
                    }
                }
                Some(Ok(Delta::Done { .. })) | None => break,
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    warn!(parent: &span, error = %e, "llm stream error");
                    break;
                }
            }
        },
        Err(e) => warn!(parent: &span, error = %e, "llm request failed"),
    }
    if !cancel.load(Ordering::Relaxed)
        && let Some(rest) = splitter.finish()
    {
        let _ = sent_tx.send(rest);
    }
    drop(decode);
    drop(ttft);
    drop(sent_tx);
    info!(parent: &span, text = %full, "reply");
    if !cancel.load(Ordering::Relaxed) {
        println!("grace: {full}");
    }
    let (spoken, any_audio) = tts_task.await.unwrap_or_default();
    // All of this reply's audio is queued: once it has played, the reply is over.
    let _ = svc.render.send(RenderCmd::EndOfTurn { turn });
    if !spoken.is_empty()
        && let Err(e) = svc.memory.add_turn("assistant", &spoken)
    {
        warn!(parent: &span, error = %e, "memory: storing reply failed");
    }
    // Learn from what Mikkel said even when he interrupted the reply: in live run 3 the turns
    // with the most facts were the interrupted ones, and they were never learned. Runs in the
    // background on the LLM's second slot, never on the voice path.
    let reply_for_memory = if spoken.is_empty() {
        "(Grace was interrupted before she could answer.)".to_owned()
    } else {
        spoken.clone()
    };
    tokio::spawn(
        svc.memory
            .clone()
            .learn(user_turn, last.1.clone(), reply_for_memory),
    );
    let _ = done.send(Done::Spoken { text: spoken });
    if !any_audio {
        let _ = done.send(Done::Silent { turn });
    }
}
