//! Turns per-frame speech probabilities into clean "speech started / stopped" edges.
//!
//! Hysteresis (a higher threshold to start than to keep going), a minimum run before speech
//! counts (coughs, clicks), and a hangover before silence counts (short pauses inside a
//! sentence). While Grace is speaking, the start threshold can be raised so leftover echo is
//! less likely to register as the user (ADR 0006 backstop).

#[derive(Debug, Clone, PartialEq)]
pub struct VadGateConfig {
    pub start_threshold: f32,
    pub stop_threshold: f32,
    /// Consecutive frames above `start_threshold` before speech is reported.
    pub min_speech_frames: u32,
    /// Consecutive frames below `stop_threshold` before silence is reported.
    pub hangover_frames: u32,
    /// Added to `start_threshold` while Grace is talking.
    pub talking_boost: f32,
    /// `min_speech_frames` while Grace is talking: residual echo comes in short blips, a real
    /// interruption lasts.
    pub talking_min_speech_frames: u32,
}

impl Default for VadGateConfig {
    fn default() -> Self {
        // 32 ms frames: 5 frames = 160 ms of speech to start, 8 frames = 256 ms of quiet to stop.
        Self {
            start_threshold: 0.5,
            stop_threshold: 0.35,
            min_speech_frames: 5,
            hangover_frames: 8,
            talking_boost: 0.3,
            talking_min_speech_frames: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEdge {
    /// Speech began at this frame (back-dated to the first frame of the run).
    Started { frame: u64 },
    /// Speech ended at this frame (back-dated to the first quiet frame).
    Stopped { frame: u64 },
}

#[derive(Debug, Clone)]
pub struct VadGate {
    cfg: VadGateConfig,
    frame: u64,
    speaking: bool,
    run: u32,
    grace_talking: bool,
}

impl VadGate {
    pub fn new(cfg: VadGateConfig) -> Self {
        Self {
            cfg,
            frame: 0,
            speaking: false,
            run: 0,
            grace_talking: false,
        }
    }

    pub fn set_grace_talking(&mut self, talking: bool) {
        self.grace_talking = talking;
    }

    pub fn speaking(&self) -> bool {
        self.speaking
    }

    pub fn push(&mut self, probability: f32) -> Option<VadEdge> {
        let frame = self.frame;
        self.frame += 1;
        if self.speaking {
            if probability < self.cfg.stop_threshold {
                self.run += 1;
                if self.run >= self.cfg.hangover_frames {
                    self.speaking = false;
                    self.run = 0;
                    return Some(VadEdge::Stopped {
                        frame: frame + 1 - u64::from(self.cfg.hangover_frames),
                    });
                }
            } else {
                self.run = 0;
            }
        } else {
            let boost = if self.grace_talking {
                self.cfg.talking_boost
            } else {
                0.0
            };
            if probability >= (self.cfg.start_threshold + boost).min(0.99) {
                self.run += 1;
                let needed = if self.grace_talking {
                    self.cfg.talking_min_speech_frames
                } else {
                    self.cfg.min_speech_frames
                };
                if self.run >= needed {
                    self.speaking = true;
                    self.run = 0;
                    return Some(VadEdge::Started {
                        frame: frame + 1 - u64::from(needed),
                    });
                }
            } else {
                self.run = 0;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(gate: &mut VadGate, probs: &[f32]) -> Vec<VadEdge> {
        probs.iter().filter_map(|&p| gate.push(p)).collect()
    }

    #[test]
    fn reports_backdated_start_and_stop() {
        let mut g = VadGate::new(VadGateConfig::default());
        let mut probs = vec![0.1; 3];
        probs.extend([0.9; 10]);
        probs.extend([0.1; 10]);
        assert_eq!(
            run(&mut g, &probs),
            vec![
                VadEdge::Started { frame: 3 },
                VadEdge::Stopped { frame: 13 }
            ]
        );
    }

    #[test]
    fn short_blips_are_ignored() {
        let mut g = VadGate::new(VadGateConfig::default());
        assert!(run(&mut g, &[0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1]).is_empty());
    }

    #[test]
    fn hysteresis_keeps_speech_through_dips_and_short_pauses() {
        let mut g = VadGate::new(VadGateConfig::default());
        let mut probs = vec![0.9; 6];
        probs.extend([0.4; 20]); // below start, above stop: still speech
        probs.extend([0.1; 5]); // pause shorter than the hangover
        probs.extend([0.9; 3]);
        let edges = run(&mut g, &probs);
        assert_eq!(edges, vec![VadEdge::Started { frame: 0 }]);
        assert!(g.speaking());
    }

    #[test]
    fn boost_while_grace_talks_rejects_moderate_echo() {
        let mut g = VadGate::new(VadGateConfig::default());
        g.set_grace_talking(true);
        assert!(run(&mut g, &[0.7; 20]).is_empty(), "0.7 < 0.5 + 0.3");
        assert!(
            run(&mut g, &[0.95; 9]).is_empty(),
            "a 288 ms blip is not an interruption"
        );
        assert!(run(&mut g, &[0.1; 1]).is_empty());
        assert_eq!(
            run(&mut g, &[0.95; 10]),
            vec![VadEdge::Started { frame: 30 }]
        );
    }
}
