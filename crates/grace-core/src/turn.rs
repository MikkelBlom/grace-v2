//! Turn-taking: who has the floor, when the user is done, and when Grace gets interrupted.
//!
//! The machine never does work itself. The engine feeds it [`Input`]s (VAD edges, turn-end
//! scores, transcripts, playback state, clock ticks) and executes the [`Command`]s it returns.
//!
//! Latency tricks live here because they are decisions about *when*, not *how*:
//! - **Speculative transcription.** STT starts the moment speech stops, in parallel with the
//!   turn-end model. If the user keeps talking, the speculative result is discarded.
//! - **Speculative replies** (`speculate`). Once a pause has a transcript but the turn is not yet
//!   committed, the reply is prepared with its audio held back ([`Command::Prepare`]); the commit
//!   then releases audio that is already there, so waiting for the turn end costs no extra time.
//!   Resuming discards it (`DiscardTranscript`).
//! - **Merging continuations** (`merge_window`). Jumping in early is cheap when it can be undone:
//!   if the user starts again before Grace's reply is audible, or within `merge_window` after it
//!   became audible, the reply is paused and the *same* turn reopens ([`Command::ReopenTurn`]).
//!   When the reopened turn ends, its audio (both parts) is transcribed again: the same words
//!   mean it was noise and the reply resumes; new words retract the reply
//!   ([`Command::RetractReply`]: it never happened) and answer the whole turn.
//! - **Ducking, then pausing, then cancelling.** Later speech during Grace's reply first ducks her
//!   output; speech that lasts `min_barge_in` pauses the reply and starts a new user turn. Only
//!   when that turn has words is the paused reply cancelled; if it had none (a printer, a cough,
//!   "hm") the reply resumes where it stopped. Live run 3 lost a reply to 3D-printer noise.
//!
//! Echo cancellation upstream is what makes "speech while Grace talks" mean the user.

use std::time::Duration;

pub type TurnId = u64;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Turn-end probability at or above which the user is considered done.
    pub turn_end_threshold: f32,
    /// Silence after which the turn ends even if the turn-end model disagrees.
    pub max_trailing_silence: Duration,
    /// Below this turn-end probability the model is confident the user is mid-thought...
    pub unfinished_threshold: f32,
    /// ...so the silence fallback waits this long instead (live run: a deliberate 2 s pause
    /// after "jeg tænker på at øh" scored 0.005 but was cut off by the 1.6 s fallback).
    pub max_trailing_silence_unfinished: Duration,
    /// Speech during a reply must last this long to count as an interruption.
    pub min_barge_in: Duration,
    /// Speech starting this soon after the reply became audible (or before it did) continues the
    /// user's turn instead of interrupting the reply. `None` turns merging off.
    pub merge_window: Option<Duration>,
    /// Prepare the reply at every pause that has a transcript, audio held until the commit.
    pub speculate: bool,
    /// Even a confident turn-end verdict waits for this much silence. With speculative replies
    /// this is free up to the time the prepared reply needs anyway, and it lets short pauses
    /// pass without jumping in.
    pub min_commit_silence: Duration,
}

impl Config {
    /// The policy before merging and speculative replies (live runs 1-4): the sim's baseline.
    pub fn without_recovery() -> Self {
        Self {
            turn_end_threshold: 0.3,
            max_trailing_silence: Duration::from_millis(1600),
            unfinished_threshold: 0.05,
            max_trailing_silence_unfinished: Duration::from_millis(3000),
            min_barge_in: Duration::from_millis(350),
            merge_window: None,
            speculate: false,
            min_commit_silence: Duration::ZERO,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            merge_window: Some(Duration::from_millis(1500)),
            speculate: true,
            ..Self::without_recovery()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    SpeechStarted,
    SpeechStopped,
    /// Turn-end model's verdict for the current pause.
    TurnEndScore {
        turn: TurnId,
        probability: f32,
    },
    /// STT result for a turn's audio up to the last pause.
    Transcript {
        turn: TurnId,
        text: String,
    },
    /// The first sample of the reply to `turn` reached the speakers.
    ReplyAudible {
        turn: TurnId,
    },
    /// Grace finished (or abandoned) speaking her reply.
    ReplyFinished {
        turn: TurnId,
    },
    /// Clock advanced; drives timeouts.
    Tick,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Start buffering audio for a new user turn.
    BeginTurn(TurnId),
    /// The user went on talking right after this turn was committed: it is open again, from
    /// its original start (the next pause re-reads and re-transcribes all of it).
    ReopenTurn(TurnId),
    /// Speech paused: ask the turn-end model and start transcribing speculatively.
    EvaluatePause(TurnId),
    /// The user resumed; any in-flight transcript (and a reply prepared from it) is stale.
    DiscardTranscript(TurnId),
    /// The turn may be over: start generating a reply to `text`, but hold its audio.
    Prepare {
        turn: TurnId,
        text: String,
    },
    /// The user is done and we have their words: reply (releasing a reply prepared from the
    /// same text, if there is one).
    Respond {
        turn: TurnId,
        text: String,
    },
    /// The turn produced no words (noise); nothing to answer.
    DropTurn(TurnId),
    DuckReply,
    UnduckReply,
    /// Someone is talking over the reply: stop playing it, but keep it until we know whether
    /// that was words.
    PauseReply(TurnId),
    /// The interruption had no words (or the reopened turn no new ones): carry on.
    ResumeReply(TurnId),
    /// The user interrupted: stop generating and speaking this reply now. What was played
    /// stays in the conversation.
    CancelReply(TurnId),
    /// The reply answered half a turn: stop it and forget it, as if it was never said.
    RetractReply(TurnId),
}

/// Words that, at the end of an utterance, mean the speaker is mid-sentence (Danish and English
/// fillers and conjunctions). From the live runs: Mikkel pauses on "øh", "og", "men", "altså".
const HANGING: &[&str] = &[
    "øh", "øhm", "hmm", "æh", "og", "men", "eller", "altså", "fordi", "så", "at", "som", "hvis",
    "når", "der", "den", "det", "til", "med", "på", "for", "om", "fx", "uh", "um", "and", "but",
    "or", "because", "so", "the", "a", "to", "with", "if", "that",
];

/// True if `text` ends on a filler or a word that can't end a sentence.
pub fn ends_unfinished(text: &str) -> bool {
    let trimmed = text.trim_end();
    // Sentence-final punctuation from the recogniser means it heard an ending.
    if trimmed.ends_with(['.', '?', '!']) {
        return false;
    }
    trimmed
        .rsplit(|c: char| !c.is_alphanumeric())
        .find(|w| !w.is_empty())
        .is_some_and(|w| HANGING.contains(&w.to_lowercase().as_str()))
}

/// Same words, ignoring case and punctuation: a re-transcription that heard nothing new.
pub fn same_words(a: &str, b: &str) -> bool {
    let words = |s: &str| {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
    };
    words(a) == words(b)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    Listening,
    UserSpeaking {
        turn: TurnId,
    },
    /// Speech paused; waiting for the turn-end verdict and/or the transcript.
    Paused {
        turn: TurnId,
        since: Duration,
        committed: bool,
        transcript: Option<String>,
        /// Silence after which this pause ends the turn regardless of the model.
        wait: Duration,
    },
    Replying {
        turn: TurnId,
        /// The transcript this reply answers.
        heard: String,
        /// Set while the user is speaking over the reply.
        overlap_since: Option<Duration>,
        /// When the reply's first sample was played (None: still silent).
        audible_since: Option<Duration>,
    },
}

/// A paused reply, waiting for the transcript of whatever paused it.
#[derive(Debug, Clone, PartialEq)]
enum Held {
    /// A new turn interrupted it.
    Interrupted(TurnId),
    /// Its own turn reopened: the user hadn't finished.
    Merging {
        turn: TurnId,
        heard: String,
        audible_since: Option<Duration>,
    },
}

#[derive(Debug)]
pub struct TurnMachine {
    cfg: Config,
    phase: Phase,
    next_turn: TurnId,
    held: Option<Held>,
}

impl TurnMachine {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            phase: Phase::Listening,
            next_turn: 1,
            held: None,
        }
    }

    pub fn phase(&self) -> &Phase {
        &self.phase
    }

    /// Applies one input at time `now` (any monotonic clock) and returns what to do.
    pub fn step(&mut self, input: Input, now: Duration) -> Vec<Command> {
        let mut out = Vec::new();
        let phase = std::mem::replace(&mut self.phase, Phase::Listening);
        self.phase = match (phase, input) {
            (Phase::Listening, Input::SpeechStarted) => self.begin_turn(&mut out),

            (Phase::UserSpeaking { turn }, Input::SpeechStopped) => {
                out.push(Command::EvaluatePause(turn));
                Phase::Paused {
                    turn,
                    since: now,
                    committed: false,
                    transcript: None,
                    wait: self.cfg.max_trailing_silence,
                }
            }

            (Phase::Paused { turn, .. }, Input::SpeechStarted) => {
                out.push(Command::DiscardTranscript(turn));
                Phase::UserSpeaking { turn }
            }

            (
                Phase::Paused {
                    turn,
                    since,
                    committed,
                    transcript,
                    wait,
                },
                Input::TurnEndScore {
                    turn: scored,
                    probability,
                },
            ) if scored == turn => {
                let committed = committed || probability >= self.cfg.turn_end_threshold;
                let wait = if probability < self.cfg.unfinished_threshold {
                    self.cfg.max_trailing_silence_unfinished
                } else {
                    wait
                };
                self.settle(turn, since, committed, transcript, wait, now, &mut out)
            }

            (
                Phase::Paused {
                    turn,
                    since,
                    committed,
                    wait,
                    ..
                },
                Input::Transcript { turn: heard, text },
            ) if heard == turn => {
                // A sentence that ends on a filler or a conjunction is not finished, whatever the
                // prosody model said ("... og øh", "... men"): wait the long timeout instead.
                let (committed, wait) = if ends_unfinished(&text) {
                    (false, wait.max(self.cfg.max_trailing_silence_unfinished))
                } else {
                    (committed, wait)
                };
                let next = self.settle(
                    turn,
                    since,
                    committed,
                    Some(text.clone()),
                    wait,
                    now,
                    &mut out,
                );
                // Not committed yet: get the reply ready, so the commit costs nothing more.
                if self.cfg.speculate
                    && matches!(next, Phase::Paused { .. })
                    && !text.trim().is_empty()
                {
                    out.push(Command::Prepare { turn, text });
                }
                next
            }

            (
                Phase::Paused {
                    turn,
                    since,
                    committed,
                    transcript,
                    wait,
                },
                Input::Tick,
            ) => self.settle(turn, since, committed, transcript, wait, now, &mut out),

            (
                Phase::Replying {
                    turn,
                    heard,
                    overlap_since: None,
                    audible_since,
                },
                Input::SpeechStarted,
            ) => {
                if self.cfg.merge_window.is_some() && audible_since.is_none() {
                    // Nothing has been said yet: he simply hadn't finished. No echo of Grace can
                    // be in the mic either, so there is nothing to wait for: reopen at once.
                    self.reopen(turn, heard, audible_since, &mut out)
                } else {
                    out.push(Command::DuckReply);
                    Phase::Replying {
                        turn,
                        heard,
                        overlap_since: Some(now),
                        audible_since,
                    }
                }
            }

            (
                Phase::Replying {
                    turn,
                    heard,
                    overlap_since: Some(_),
                    audible_since,
                },
                Input::SpeechStopped,
            ) => {
                out.push(Command::UnduckReply);
                Phase::Replying {
                    turn,
                    heard,
                    overlap_since: None,
                    audible_since,
                }
            }

            (
                Phase::Replying {
                    turn,
                    heard,
                    overlap_since: Some(since),
                    audible_since,
                },
                Input::Tick,
            ) if now.saturating_sub(since) >= self.cfg.min_barge_in => {
                // Within the merge window this is the rest of his turn, not an interruption.
                let continues = match (self.cfg.merge_window, audible_since) {
                    (Some(window), Some(audible)) => since.saturating_sub(audible) < window,
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                if continues {
                    self.reopen(turn, heard, audible_since, &mut out)
                } else {
                    out.push(Command::PauseReply(turn));
                    self.held = Some(Held::Interrupted(turn));
                    self.begin_turn(&mut out)
                }
            }

            (
                Phase::Replying {
                    turn,
                    heard,
                    overlap_since,
                    audible_since: None,
                },
                Input::ReplyAudible { turn: played },
            ) if played == turn => Phase::Replying {
                turn,
                heard,
                overlap_since,
                audible_since: Some(now),
            },

            (
                Phase::Replying {
                    turn,
                    overlap_since,
                    ..
                },
                Input::ReplyFinished { turn: done },
            ) if done == turn => {
                if overlap_since.is_some() {
                    // The reply ended while the user was already talking: that speech is a turn.
                    out.push(Command::UnduckReply);
                    self.begin_turn(&mut out)
                } else {
                    Phase::Listening
                }
            }

            // Everything else (stale ids, duplicate edges, ticks with nothing due) is a no-op.
            (phase, _) => phase,
        };
        out
    }

    fn begin_turn(&mut self, out: &mut Vec<Command>) -> Phase {
        let turn = self.next_turn;
        self.next_turn += 1;
        out.push(Command::BeginTurn(turn));
        Phase::UserSpeaking { turn }
    }

    /// The user wasn't done: hold the reply and open the same turn again.
    fn reopen(
        &mut self,
        turn: TurnId,
        heard: String,
        audible_since: Option<Duration>,
        out: &mut Vec<Command>,
    ) -> Phase {
        out.push(Command::PauseReply(turn));
        out.push(Command::ReopenTurn(turn));
        self.held = Some(Held::Merging {
            turn,
            heard,
            audible_since,
        });
        Phase::UserSpeaking { turn }
    }

    /// Moves to a reply once the turn is both over and transcribed: the turn-end model said so
    /// (`committed`) and the silence has lasted `min_commit_silence`, or the silence outlasted
    /// `wait`.
    #[allow(clippy::too_many_arguments)] // the fields of `Phase::Paused`, plus the clock
    fn settle(
        &mut self,
        turn: TurnId,
        since: Duration,
        committed: bool,
        transcript: Option<String>,
        wait: Duration,
        now: Duration,
        out: &mut Vec<Command>,
    ) -> Phase {
        let silence = now.saturating_sub(since);
        let over = (committed && silence >= self.cfg.min_commit_silence) || silence >= wait;
        let text = match (over, transcript) {
            (true, Some(text)) => text,
            (_, transcript) => {
                return Phase::Paused {
                    turn,
                    since,
                    committed,
                    transcript,
                    wait,
                };
            }
        };
        let empty = text.trim().is_empty();
        let replying = |turn, heard, audible_since| Phase::Replying {
            turn,
            heard,
            overlap_since: None,
            audible_since,
        };
        match self.held.take() {
            // The reopened turn added nothing (a cough, a breath): the held reply still fits.
            Some(Held::Merging {
                turn: held,
                heard,
                audible_since,
            }) if held == turn && (empty || same_words(&text, &heard)) => {
                out.push(Command::ResumeReply(held));
                replying(held, heard, audible_since)
            }
            Some(Held::Merging { turn: held, .. }) => {
                out.push(Command::RetractReply(held));
                out.push(Command::Respond {
                    turn,
                    text: text.clone(),
                });
                replying(turn, text, None)
            }
            // A false interruption: Grace carries on (long past any merge window).
            Some(Held::Interrupted(held)) if empty => {
                out.push(Command::DropTurn(turn));
                out.push(Command::ResumeReply(held));
                replying(held, String::new(), Some(Duration::ZERO))
            }
            Some(Held::Interrupted(held)) => {
                out.push(Command::CancelReply(held));
                out.push(Command::Respond {
                    turn,
                    text: text.clone(),
                });
                replying(turn, text, None)
            }
            None if empty => {
                out.push(Command::DropTurn(turn));
                Phase::Listening
            }
            None => {
                out.push(Command::Respond {
                    turn,
                    text: text.clone(),
                });
                replying(turn, text, None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    fn machine() -> TurnMachine {
        TurnMachine::new(Config::without_recovery())
    }

    #[test]
    fn happy_path_turn_end_then_transcript() {
        let mut m = machine();
        assert_eq!(
            m.step(Input::SpeechStarted, ms(0)),
            vec![Command::BeginTurn(1)]
        );
        assert_eq!(
            m.step(Input::SpeechStopped, ms(900)),
            vec![Command::EvaluatePause(1)]
        );
        let score = Input::TurnEndScore {
            turn: 1,
            probability: 0.9,
        };
        assert_eq!(m.step(score, ms(1000)), vec![]);
        let heard = Input::Transcript {
            turn: 1,
            text: "hej grace".into(),
        };
        assert_eq!(
            m.step(heard, ms(1050)),
            vec![Command::Respond {
                turn: 1,
                text: "hej grace".into()
            }]
        );
        assert_eq!(m.step(Input::ReplyFinished { turn: 1 }, ms(3000)), vec![]);
        assert_eq!(m.phase(), &Phase::Listening);
    }

    #[test]
    fn transcript_before_turn_end_waits_for_commit() {
        let mut m = machine();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(500));
        let heard = Input::Transcript {
            turn: 1,
            text: "og så".into(),
        };
        assert_eq!(m.step(heard, ms(600)), vec![]);
        let score = Input::TurnEndScore {
            turn: 1,
            probability: 0.8,
        };
        assert_eq!(
            m.step(score, ms(650)),
            vec![Command::Respond {
                turn: 1,
                text: "og så".into()
            }]
        );
    }

    #[test]
    fn low_turn_end_score_falls_back_to_silence_timeout() {
        let mut m = machine();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.1,
            },
            ms(1100),
        );
        m.step(
            Input::Transcript {
                turn: 1,
                text: "jeg tænker meget".into(),
            },
            ms(1150),
        );
        assert_eq!(m.step(Input::Tick, ms(2000)), vec![], "not yet due");
        assert_eq!(
            m.step(Input::Tick, ms(2600)),
            vec![Command::Respond {
                turn: 1,
                text: "jeg tænker meget".into()
            }]
        );
    }

    #[test]
    fn resuming_speech_discards_speculative_transcript() {
        let mut m = machine();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(500));
        assert_eq!(
            m.step(Input::SpeechStarted, ms(700)),
            vec![Command::DiscardTranscript(1)]
        );
        // The stale transcript from the first pause arrives late and must be ignored...
        // ...but the machine is in UserSpeaking, so it is simply a no-op.
        assert_eq!(
            m.step(
                Input::Transcript {
                    turn: 1,
                    text: "halv".into()
                },
                ms(750)
            ),
            vec![]
        );
        assert_eq!(m.phase(), &Phase::UserSpeaking { turn: 1 });
    }

    #[test]
    fn empty_transcript_drops_the_turn() {
        let mut m = machine();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(300));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 1.0,
            },
            ms(350),
        );
        assert_eq!(
            m.step(
                Input::Transcript {
                    turn: 1,
                    text: "  ".into()
                },
                ms(400)
            ),
            vec![Command::DropTurn(1)]
        );
        assert_eq!(m.phase(), &Phase::Listening);
    }

    fn replying(m: &mut TurnMachine) {
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(300));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 1.0,
            },
            ms(320),
        );
        m.step(
            Input::Transcript {
                turn: 1,
                text: "fortæl en historie".into(),
            },
            ms(340),
        );
    }

    #[test]
    fn short_overlap_ducks_then_unducks() {
        let mut m = machine();
        replying(&mut m);
        assert_eq!(
            m.step(Input::SpeechStarted, ms(2000)),
            vec![Command::DuckReply]
        );
        assert_eq!(m.step(Input::Tick, ms(2100)), vec![]);
        assert_eq!(
            m.step(Input::SpeechStopped, ms(2150)),
            vec![Command::UnduckReply]
        );
        assert!(matches!(m.phase(), Phase::Replying { turn: 1, .. }));
    }

    #[test]
    fn sustained_overlap_pauses_and_starts_new_turn() {
        let mut m = machine();
        replying(&mut m);
        m.step(Input::SpeechStarted, ms(2000));
        assert_eq!(
            m.step(Input::Tick, ms(2400)),
            vec![Command::PauseReply(1), Command::BeginTurn(2)]
        );
        // The paused reply reporting completion afterwards must not disturb turn 2.
        assert_eq!(m.step(Input::ReplyFinished { turn: 1 }, ms(2450)), vec![]);
        assert_eq!(m.phase(), &Phase::UserSpeaking { turn: 2 });
    }

    /// Interrupts reply 1 and finishes the interrupting turn 2 with `text`.
    fn interrupt_with(m: &mut TurnMachine, text: &str) -> Vec<Command> {
        replying(m);
        m.step(Input::SpeechStarted, ms(2000));
        m.step(Input::Tick, ms(2400));
        m.step(Input::SpeechStopped, ms(2600));
        m.step(
            Input::TurnEndScore {
                turn: 2,
                probability: 0.9,
            },
            ms(2650),
        );
        m.step(
            Input::Transcript {
                turn: 2,
                text: text.into(),
            },
            ms(2700),
        )
    }

    #[test]
    fn interruption_with_words_cancels_the_paused_reply() {
        let mut m = machine();
        let out = interrupt_with(&mut m, "vent lige");
        assert_eq!(
            out,
            vec![
                Command::CancelReply(1),
                Command::Respond {
                    turn: 2,
                    text: "vent lige".into()
                }
            ]
        );
    }

    #[test]
    fn interruption_without_words_resumes_the_reply() {
        // Live run 3: 3D-printer noise cut Grace off and she never answered.
        let mut m = machine();
        let out = interrupt_with(&mut m, "");
        assert_eq!(out, vec![Command::DropTurn(2), Command::ResumeReply(1)]);
        assert!(matches!(m.phase(), Phase::Replying { turn: 1, .. }));
        assert_eq!(m.step(Input::ReplyFinished { turn: 1 }, ms(5000)), vec![]);
        assert_eq!(m.phase(), &Phase::Listening);
    }

    #[test]
    fn stale_scores_for_other_turns_are_ignored() {
        let mut m = machine();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(300));
        assert_eq!(
            m.step(
                Input::TurnEndScore {
                    turn: 99,
                    probability: 1.0
                },
                ms(310)
            ),
            vec![]
        );
        assert!(matches!(
            m.phase(),
            Phase::Paused {
                committed: false,
                ..
            }
        ));
    }

    #[test]
    fn confident_unfinished_score_waits_longer() {
        let mut m = machine();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::Transcript {
                turn: 1,
                text: "jeg tænker på at øh".into(),
            },
            ms(1100),
        );
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.005,
            },
            ms(1150),
        );
        assert_eq!(
            m.step(Input::Tick, ms(2700)),
            vec![],
            "1.7 s: still waiting"
        );
        assert_eq!(
            m.step(Input::Tick, ms(3900)),
            vec![],
            "2.9 s: still waiting"
        );
        assert_eq!(
            m.step(Input::Tick, ms(4000)),
            vec![Command::Respond {
                turn: 1,
                text: "jeg tænker på at øh".into()
            }]
        );
    }

    #[test]
    fn hanging_words_mean_unfinished() {
        assert!(ends_unfinished("min søster hun bor i Caribien nu og øh"));
        assert!(ends_unfinished("jeg tænker på at"));
        assert!(ends_unfinished("det er fordi"));
        assert!(!ends_unfinished("hvor bor min søster"));
        assert!(!ends_unfinished("Det er det."));
        assert!(!ends_unfinished("er det så?"));
        assert!(!ends_unfinished(""));
    }

    #[test]
    fn hanging_transcript_overrides_a_confident_turn_end() {
        let mut m = machine();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.97,
            },
            ms(1050),
        );
        assert_eq!(
            m.step(
                Input::Transcript {
                    turn: 1,
                    text: "og så skal vi øh".into()
                },
                ms(1100)
            ),
            vec![],
            "waits instead of replying"
        );
        assert_eq!(
            m.step(Input::Tick, ms(2700)),
            vec![],
            "past the normal 1.6 s fallback"
        );
        assert!(matches!(
            m.step(Input::Tick, ms(4100))[..],
            [Command::Respond { turn: 1, .. }]
        ));
    }

    // --- Merging continuations and speculative replies -----------------------------------

    fn recovering() -> TurnMachine {
        TurnMachine::new(Config::default())
    }

    /// Turn 1 ("hvad med i morgen") committed on a confident score; Grace is replying.
    fn committed(m: &mut TurnMachine) {
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.9,
            },
            ms(1050),
        );
        let out = m.step(
            Input::Transcript {
                turn: 1,
                text: "hvad med i morgen".into(),
            },
            ms(1300),
        );
        assert_eq!(
            out,
            vec![Command::Respond {
                turn: 1,
                text: "hvad med i morgen".into()
            }]
        );
    }

    /// The reopened turn 1 pauses again and is scored and transcribed as `text`.
    fn finish_reopened(m: &mut TurnMachine, text: &str, at: u64) -> Vec<Command> {
        m.step(Input::SpeechStopped, ms(at));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.9,
            },
            ms(at + 50),
        );
        m.step(
            Input::Transcript {
                turn: 1,
                text: text.into(),
            },
            ms(at + 300),
        )
    }

    #[test]
    fn resuming_before_grace_is_heard_reopens_the_same_turn() {
        let mut m = recovering();
        committed(&mut m);
        // No duck-and-wait: nothing is playing yet, so this can only be him.
        assert_eq!(
            m.step(Input::SpeechStarted, ms(1600)),
            vec![Command::PauseReply(1), Command::ReopenTurn(1)]
        );
        assert_eq!(m.phase(), &Phase::UserSpeaking { turn: 1 });
        // A stale "audible" from the held reply changes nothing.
        assert_eq!(m.step(Input::ReplyAudible { turn: 1 }, ms(1700)), vec![]);
        assert_eq!(
            finish_reopened(&mut m, "hvad med i morgen, kan du tjekke vejret", 3000),
            vec![
                Command::RetractReply(1),
                Command::Respond {
                    turn: 1,
                    text: "hvad med i morgen, kan du tjekke vejret".into()
                }
            ]
        );
    }

    #[test]
    fn reopened_turn_with_no_new_words_resumes_the_reply() {
        let mut m = recovering();
        committed(&mut m);
        m.step(Input::SpeechStarted, ms(1600));
        // A cough: the re-transcription hears the same words (punctuation aside).
        assert_eq!(
            finish_reopened(&mut m, "Hvad med i morgen?", 2000),
            vec![Command::ResumeReply(1)]
        );
        assert!(matches!(m.phase(), Phase::Replying { turn: 1, .. }));
        assert_eq!(m.step(Input::ReplyFinished { turn: 1 }, ms(6000)), vec![]);
        assert_eq!(m.phase(), &Phase::Listening);
    }

    #[test]
    fn resuming_inside_the_merge_window_reopens_after_min_barge_in() {
        let mut m = recovering();
        committed(&mut m);
        m.step(Input::ReplyAudible { turn: 1 }, ms(2100));
        // 1.0 s after Grace became audible: within the 1.5 s window, but she is audible now,
        // so it must last min_barge_in first (it could be her echo).
        assert_eq!(
            m.step(Input::SpeechStarted, ms(3100)),
            vec![Command::DuckReply]
        );
        assert_eq!(m.step(Input::Tick, ms(3300)), vec![]);
        assert_eq!(
            m.step(Input::Tick, ms(3460)),
            vec![Command::PauseReply(1), Command::ReopenTurn(1)]
        );
        assert!(matches!(
            finish_reopened(&mut m, "hvad med i morgen eftermiddag", 4500)[..],
            [Command::RetractReply(1), Command::Respond { turn: 1, .. }]
        ));
    }

    #[test]
    fn speech_after_the_merge_window_is_a_new_turn() {
        let mut m = recovering();
        committed(&mut m);
        m.step(Input::ReplyAudible { turn: 1 }, ms(2100));
        m.step(Input::SpeechStarted, ms(3700));
        assert_eq!(
            m.step(Input::Tick, ms(4100)),
            vec![Command::PauseReply(1), Command::BeginTurn(2)]
        );
        m.step(Input::SpeechStopped, ms(4600));
        m.step(
            Input::TurnEndScore {
                turn: 2,
                probability: 0.9,
            },
            ms(4650),
        );
        assert_eq!(
            m.step(
                Input::Transcript {
                    turn: 2,
                    text: "nej vent".into()
                },
                ms(4900)
            ),
            vec![
                Command::CancelReply(1),
                Command::Respond {
                    turn: 2,
                    text: "nej vent".into()
                }
            ]
        );
    }

    #[test]
    fn without_merging_resumed_speech_still_ducks() {
        let mut m = TurnMachine::new(Config {
            merge_window: None,
            ..Config::default()
        });
        committed(&mut m);
        assert_eq!(
            m.step(Input::SpeechStarted, ms(1600)),
            vec![Command::DuckReply]
        );
    }

    #[test]
    fn an_uncommitted_pause_prepares_a_reply() {
        let mut m = recovering();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.1,
            },
            ms(1050),
        );
        assert_eq!(
            m.step(
                Input::Transcript {
                    turn: 1,
                    text: "jeg tænker meget".into()
                },
                ms(1300)
            ),
            vec![Command::Prepare {
                turn: 1,
                text: "jeg tænker meget".into()
            }]
        );
        // The fallback commits the same text: Respond releases the prepared reply.
        assert_eq!(
            m.step(Input::Tick, ms(2600)),
            vec![Command::Respond {
                turn: 1,
                text: "jeg tænker meget".into()
            }]
        );
    }

    #[test]
    fn resuming_discards_the_prepared_reply() {
        let mut m = recovering();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::Transcript {
                turn: 1,
                text: "jeg tænker".into(),
            },
            ms(1300),
        );
        assert_eq!(
            m.step(Input::SpeechStarted, ms(1500)),
            vec![Command::DiscardTranscript(1)]
        );
    }

    #[test]
    fn a_committed_pause_or_empty_transcript_prepares_nothing() {
        let mut m = recovering();
        committed(&mut m); // asserts Respond alone
        let mut m = recovering();
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(300));
        assert_eq!(
            m.step(
                Input::Transcript {
                    turn: 1,
                    text: " ".into()
                },
                ms(500)
            ),
            vec![]
        );
    }

    #[test]
    fn a_confident_verdict_still_waits_for_min_commit_silence() {
        let mut m = TurnMachine::new(Config {
            min_commit_silence: ms(600),
            ..Config::default()
        });
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.9,
            },
            ms(1050),
        );
        // 300 ms of silence: get the reply ready, but don't jump in yet.
        assert_eq!(
            m.step(
                Input::Transcript {
                    turn: 1,
                    text: "hej grace".into()
                },
                ms(1300)
            ),
            vec![Command::Prepare {
                turn: 1,
                text: "hej grace".into()
            }]
        );
        assert_eq!(m.step(Input::Tick, ms(1590)), vec![]);
        assert_eq!(
            m.step(Input::Tick, ms(1600)),
            vec![Command::Respond {
                turn: 1,
                text: "hej grace".into()
            }]
        );
    }

    #[test]
    fn a_pause_shorter_than_min_commit_silence_never_commits() {
        let mut m = TurnMachine::new(Config {
            min_commit_silence: ms(600),
            ..Config::default()
        });
        m.step(Input::SpeechStarted, ms(0));
        m.step(Input::SpeechStopped, ms(1000));
        m.step(
            Input::TurnEndScore {
                turn: 1,
                probability: 0.9,
            },
            ms(1050),
        );
        m.step(
            Input::Transcript {
                turn: 1,
                text: "hej grace".into(),
            },
            ms(1300),
        );
        assert_eq!(
            m.step(Input::SpeechStarted, ms(1500)),
            vec![Command::DiscardTranscript(1)]
        );
        assert_eq!(m.phase(), &Phase::UserSpeaking { turn: 1 });
    }

    #[test]
    fn same_words_ignores_case_and_punctuation() {
        assert!(same_words("Hvad med i morgen?", "hvad med i morgen"));
        assert!(!same_words("hvad med i morgen", "hvad med i morgen aften"));
        assert!(same_words("", "  "));
    }
}
