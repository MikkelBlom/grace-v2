//! Turn-taking: who has the floor, when the user is done, and when Grace gets interrupted.
//!
//! The machine never does work itself. The engine feeds it [`Input`]s (VAD edges, turn-end
//! scores, transcripts, playback state, clock ticks) and executes the [`Command`]s it returns.
//!
//! Two latency tricks live here because they are decisions about *when*, not *how*:
//! - **Speculative transcription.** STT starts the moment speech stops, in parallel with the
//!   turn-end model. If the user keeps talking, the speculative result is discarded.
//! - **Ducking, then pausing, then cancelling.** Speech during Grace's reply first ducks her
//!   output; speech that lasts `min_barge_in` pauses the reply and starts a user turn. Only when
//!   that turn has words is the paused reply cancelled; if it had none (a printer, a cough, "hm")
//!   the reply resumes where it stopped. Live run 3 lost a reply to 3D-printer noise.
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Smart Turn da1 on CoRal validation speakers: at 0.3 it notices 84 % of finished
            // turns and cuts off 31 % of unfinished ones; v3.2 at 0.5 managed 64 % / 40 %.
            turn_end_threshold: 0.3,
            max_trailing_silence: Duration::from_millis(1600),
            unfinished_threshold: 0.05,
            max_trailing_silence_unfinished: Duration::from_millis(3000),
            min_barge_in: Duration::from_millis(350),
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
    /// Speech paused: ask the turn-end model and start transcribing speculatively.
    EvaluatePause(TurnId),
    /// The user resumed; any in-flight transcript for this turn is stale.
    DiscardTranscript(TurnId),
    /// The user is done and we have their words: produce a reply.
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
    /// The interruption had no words: carry on with the paused reply.
    ResumeReply(TurnId),
    /// The user interrupted: stop generating and speaking this reply now.
    CancelReply(TurnId),
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
        /// Set while the user is speaking over the reply.
        overlap_since: Option<Duration>,
    },
}

#[derive(Debug)]
pub struct TurnMachine {
    cfg: Config,
    phase: Phase,
    next_turn: TurnId,
    /// A reply paused by an interruption, waiting for that interruption's transcript.
    held: Option<TurnId>,
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
                self.settle(turn, since, committed, transcript, wait, &mut out)
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
                self.settle(turn, since, committed, Some(text), wait, &mut out)
            }

            (
                Phase::Paused {
                    turn,
                    since,
                    transcript,
                    wait,
                    ..
                },
                Input::Tick,
            ) if now.saturating_sub(since) >= wait => {
                self.settle(turn, since, true, transcript, wait, &mut out)
            }

            (
                Phase::Replying {
                    turn,
                    overlap_since: None,
                },
                Input::SpeechStarted,
            ) => {
                out.push(Command::DuckReply);
                Phase::Replying {
                    turn,
                    overlap_since: Some(now),
                }
            }

            (
                Phase::Replying {
                    turn,
                    overlap_since: Some(_),
                },
                Input::SpeechStopped,
            ) => {
                out.push(Command::UnduckReply);
                Phase::Replying {
                    turn,
                    overlap_since: None,
                }
            }

            (
                Phase::Replying {
                    turn,
                    overlap_since: Some(since),
                },
                Input::Tick,
            ) if now.saturating_sub(since) >= self.cfg.min_barge_in => {
                out.push(Command::PauseReply(turn));
                self.held = Some(turn);
                self.begin_turn(&mut out)
            }

            (
                Phase::Replying {
                    turn,
                    overlap_since,
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

    /// Moves to a reply once the turn is both committed and transcribed.
    fn settle(
        &mut self,
        turn: TurnId,
        since: Duration,
        committed: bool,
        transcript: Option<String>,
        wait: Duration,
        out: &mut Vec<Command>,
    ) -> Phase {
        match (committed, transcript) {
            (true, Some(text)) if text.trim().is_empty() => {
                out.push(Command::DropTurn(turn));
                match self.held.take() {
                    // A false interruption: Grace carries on.
                    Some(held) => {
                        out.push(Command::ResumeReply(held));
                        Phase::Replying {
                            turn: held,
                            overlap_since: None,
                        }
                    }
                    None => Phase::Listening,
                }
            }
            (true, Some(text)) => {
                if let Some(held) = self.held.take() {
                    out.push(Command::CancelReply(held));
                }
                out.push(Command::Respond { turn, text });
                Phase::Replying {
                    turn,
                    overlap_since: None,
                }
            }
            (committed, transcript) => Phase::Paused {
                turn,
                since,
                committed,
                transcript,
                wait,
            },
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
        TurnMachine::new(Config::default())
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
}
