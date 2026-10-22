//! What the overlay shows, as pure state over an explicit clock (milliseconds): the glow's fade
//! and gentle pulse, and focus boxes that fade in, live for their `ms` and fade out.

use crate::protocol::{BoxSpec, Command, Rect};

pub const GLOW_FADE_MS: f32 = 450.0;
/// One breath of the glow while Grace works; the pulse dips to 1 - PULSE_DEPTH.
pub const PULSE_MS: f32 = 2400.0;
pub const PULSE_DEPTH: f32 = 0.25;
pub const BOX_FADE_IN_MS: u64 = 150;
pub const BOX_FADE_OUT_MS: u64 = 350;
pub const MAX_BOXES: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct FocusBox {
    pub rect: Rect,
    pub label: Option<String>,
    pub born_ms: u64,
    /// When it starts fading out.
    pub until_ms: u64,
}

impl FocusBox {
    pub fn alpha(&self, now_ms: u64) -> f32 {
        let fade_in = (now_ms.saturating_sub(self.born_ms) as f32 / BOX_FADE_IN_MS as f32).min(1.0);
        let fade_out = if now_ms <= self.until_ms {
            1.0
        } else {
            1.0 - (now_ms - self.until_ms) as f32 / BOX_FADE_OUT_MS as f32
        };
        fade_in.min(fade_out).clamp(0.0, 1.0)
    }
    fn gone(&self, now_ms: u64) -> bool {
        now_ms >= self.until_ms + BOX_FADE_OUT_MS
    }
}

#[derive(Debug, Default)]
pub struct Scene {
    pub glow_on: bool,
    /// 0..1, eased toward glow_on.
    glow_level: f32,
    glow_since_ms: u64,
    last_ms: u64,
    pub boxes: Vec<FocusBox>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies a command; Ping and Quit are handled by the caller.
    pub fn apply(&mut self, cmd: &Command, now_ms: u64) {
        self.tick(now_ms);
        match cmd {
            Command::Glow(on) => {
                if *on && !self.glow_on {
                    self.glow_since_ms = now_ms;
                }
                self.glow_on = *on;
            }
            Command::Box(BoxSpec { rect, label, ms }) => {
                // The same spot pointed at again replaces the old box instead of stacking.
                self.boxes.retain(|b| b.rect != *rect);
                self.boxes.push(FocusBox {
                    rect: *rect,
                    label: label.clone(),
                    born_ms: now_ms,
                    until_ms: now_ms + ms,
                });
                if self.boxes.len() > MAX_BOXES {
                    self.boxes.remove(0);
                }
            }
            Command::Clear => {
                for b in &mut self.boxes {
                    b.until_ms = b.until_ms.min(now_ms);
                }
            }
            Command::Ping | Command::Quit => {}
        }
    }

    /// Advances the fades. Returns true while anything is visible or still animating, i.e.
    /// while the overlay needs frames.
    pub fn tick(&mut self, now_ms: u64) -> bool {
        let dt = now_ms.saturating_sub(self.last_ms) as f32;
        self.last_ms = self.last_ms.max(now_ms);
        let step = dt / GLOW_FADE_MS;
        self.glow_level = if self.glow_on {
            (self.glow_level + step).min(1.0)
        } else {
            (self.glow_level - step).max(0.0)
        };
        self.boxes.retain(|b| !b.gone(now_ms));
        self.visible()
    }

    /// True while something fades (needs smooth frames); false once only the slow breath and
    /// steady boxes remain, which a low frame rate draws just as well at a fraction of the CPU.
    pub fn fading(&self, now_ms: u64) -> bool {
        let glow_moving = self.glow_level > 0.0 && self.glow_level < 1.0;
        glow_moving || self.boxes.iter().any(|b| b.alpha(now_ms) < 1.0)
    }

    pub fn visible(&self) -> bool {
        self.glow_level > 0.0 || self.glow_on || !self.boxes.is_empty()
    }

    /// The glow's opacity now: the eased fade times a slow breath.
    pub fn glow_alpha(&self, now_ms: u64) -> f32 {
        let ease = self.glow_level * self.glow_level * (3.0 - 2.0 * self.glow_level);
        let phase =
            (now_ms.saturating_sub(self.glow_since_ms) as f32 / PULSE_MS) * std::f32::consts::TAU;
        let breath = 1.0 - PULSE_DEPTH * 0.5 * (1.0 - phase.cos());
        ease * breath
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(x: i32, ms: u64) -> Command {
        Command::Box(BoxSpec {
            rect: Rect::new(x, 0, 10, 10),
            label: None,
            ms,
        })
    }

    #[test]
    fn glow_fades_in_and_out() {
        let mut s = Scene::new();
        assert!(!s.tick(0));
        s.apply(&Command::Glow(true), 0);
        assert!(s.tick(0));
        assert_eq!(s.glow_alpha(0), 0.0);
        s.tick(200);
        let mid = s.glow_alpha(200);
        assert!(mid > 0.0 && mid < 1.0, "{mid}");
        assert!(s.fading(200));
        s.tick(1000);
        assert!(!s.fading(1000));
        // Full level; the pulse starts at its peak.
        assert!((s.glow_alpha(s.glow_since_ms) - 1.0).abs() < 1e-6);
        assert!(s.glow_alpha(1200) >= 1.0 - PULSE_DEPTH - 1e-6);
        s.apply(&Command::Glow(false), 1000);
        assert!(s.tick(1200));
        assert!(!s.tick(2000));
        assert_eq!(s.glow_alpha(2000), 0.0);
    }

    #[test]
    fn boxes_live_their_time_then_fade() {
        let mut s = Scene::new();
        s.apply(&bx(0, 1000), 0);
        assert_eq!(s.boxes[0].alpha(0), 0.0);
        assert_eq!(s.boxes[0].alpha(BOX_FADE_IN_MS), 1.0);
        assert_eq!(s.boxes[0].alpha(1000), 1.0);
        assert!(s.fading(10) && !s.fading(500));
        assert!(s.tick(1000 + BOX_FADE_OUT_MS / 2));
        assert!(s.boxes[0].alpha(1000 + BOX_FADE_OUT_MS / 2) < 1.0);
        assert!(!s.tick(1000 + BOX_FADE_OUT_MS));
        assert!(s.boxes.is_empty());
    }

    #[test]
    fn clear_fades_every_box_and_keeps_the_glow() {
        let mut s = Scene::new();
        s.apply(&Command::Glow(true), 0);
        s.apply(&bx(0, 5000), 0);
        s.apply(&bx(50, 5000), 0);
        s.apply(&Command::Clear, 100);
        s.tick(100 + BOX_FADE_OUT_MS);
        assert!(s.boxes.is_empty());
        assert!(s.glow_on && s.visible());
    }

    #[test]
    fn same_spot_replaces_and_count_is_capped() {
        let mut s = Scene::new();
        s.apply(&bx(0, 1000), 0);
        s.apply(&bx(0, 3000), 10);
        assert_eq!(s.boxes.len(), 1);
        assert_eq!(s.boxes[0].until_ms, 3010);
        for i in 0..20 {
            s.apply(&bx(100 + i, 1000), 20);
        }
        assert_eq!(s.boxes.len(), MAX_BOXES);
        assert_eq!(s.boxes.last().unwrap().rect.x, 119);
    }
}
