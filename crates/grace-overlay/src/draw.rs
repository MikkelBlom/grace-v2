//! CPU drawing into a premultiplied BGRA buffer (what UpdateLayeredWindow takes): the edge glow,
//! focus boxes and label pills. No GPU, so it looks the same on either GPU of a hybrid laptop.

use crate::protocol::Rect;

/// Claude's computer-use orange.
pub const ORANGE: (u8, u8, u8) = (0xE8, 0x76, 0x3A);
pub const WHITE: (u8, u8, u8) = (0xFF, 0xFF, 0xFF);
/// Glow band width and box line width at 96 DPI; scaled by the monitor's DPI.
pub const GLOW_PX: i32 = 30;
pub const BOX_LINE_PX: i32 = 3;
pub const BOX_HALO_PX: i32 = 8;
pub const LABEL_PAD_PX: i32 = 6;
const GLOW_PEAK: f32 = 0.9;
const BOX_FILL: f32 = 0.08;

pub fn premul((r, g, b): (u8, u8, u8), a: f32) -> u32 {
    let a = a.clamp(0.0, 1.0);
    let m = |c: u8| (c as f32 * a + 0.5) as u32;
    ((a * 255.0 + 0.5) as u32) << 24 | m(r) << 16 | m(g) << 8 | m(b)
}

/// (r, g, b, a) of a premultiplied pixel, for tests and checks.
pub fn unpack(p: u32) -> (u8, u8, u8, u8) {
    ((p >> 16) as u8, (p >> 8) as u8, p as u8, (p >> 24) as u8)
}

/// Source-over for premultiplied pixels.
fn over(dst: u32, src: u32) -> u32 {
    let sa = src >> 24;
    if sa == 255 {
        return src;
    }
    if sa == 0 {
        return dst;
    }
    let inv = 255 - sa;
    let ch = |shift: u32| {
        let s = (src >> shift) & 0xFF;
        let d = (dst >> shift) & 0xFF;
        (s + (d * inv + 127) / 255).min(255) << shift
    };
    ch(24) | ch(16) | ch(8) | ch(0)
}

pub struct Canvas<'a> {
    pub px: &'a mut [u32],
    pub w: i32,
    pub h: i32,
}

impl<'a> Canvas<'a> {
    pub fn new(px: &'a mut [u32], w: i32, h: i32) -> Self {
        assert_eq!(px.len(), (w * h) as usize);
        Self { px, w, h }
    }

    pub fn get(&self, x: i32, y: i32) -> u32 {
        self.px[(y * self.w + x) as usize]
    }

    pub fn clear(&mut self) {
        self.px.fill(0);
    }

    fn clip(&self, r: Rect) -> Option<(i32, i32, i32, i32)> {
        let (x0, y0) = (r.x.max(0), r.y.max(0));
        let (x1, y1) = (r.right().min(self.w), r.bottom().min(self.h));
        (x0 < x1 && y0 < y1).then_some((x0, y0, x1, y1))
    }

    pub fn fill(&mut self, r: Rect, color: (u8, u8, u8), alpha: f32) {
        let src = premul(color, alpha);
        let Some((x0, y0, x1, y1)) = self.clip(r) else {
            return;
        };
        for y in y0..y1 {
            let row = (y * self.w) as usize;
            for p in &mut self.px[row + x0 as usize..row + x1 as usize] {
                *p = over(*p, src);
            }
        }
    }

    /// The orange glow along this canvas's edges: strongest at the edge, gone `band` pixels in.
    /// Draw it first: it sets the band pixels instead of blending (a blend cost 2.9 ms per
    /// 2560x1600 frame), and touches only the bands, so the interior is left as it was.
    pub fn glow(&mut self, band: i32, alpha: f32) {
        if band <= 0 {
            return;
        }
        let band = band.min(self.w / 2).min(self.h / 2).max(1);
        // Quadratic falloff reads as a soft light, not a stripe.
        let lut: Vec<u32> = (0..band)
            .map(|d| {
                let t = 1.0 - d as f32 / band as f32;
                premul(ORANGE, GLOW_PEAK * alpha * t * t)
            })
            .collect();
        let (w, h) = (self.w, self.h);
        for y in 0..h {
            let dy = y.min(h - 1 - y);
            let row = (y * w) as usize;
            let put = |x: i32, px: &mut [u32]| {
                let d = dy.min(x).min(w - 1 - x);
                if d < band {
                    px[row + x as usize] = lut[d as usize];
                }
            };
            if dy < band {
                for x in 0..w {
                    put(x, self.px);
                }
            } else {
                for x in (0..band).chain(w - band..w) {
                    put(x, self.px);
                }
            }
        }
    }

    /// A focus box: a soft halo, a faint fill and a crisp outline.
    pub fn focus_box(&mut self, r: Rect, scale: f32, alpha: f32) {
        let line = ((BOX_LINE_PX as f32 * scale).round() as i32).max(1);
        let halo = ((BOX_HALO_PX as f32 * scale).round() as i32).max(1);
        for i in 1..=halo {
            let t = 1.0 - i as f32 / (halo + 1) as f32;
            self.outline(r.inflate(line + i - 1), 1, ORANGE, 0.35 * alpha * t * t);
        }
        self.fill(r, ORANGE, BOX_FILL * alpha);
        self.outline(r.inflate(line), line, ORANGE, alpha);
    }

    /// A `thick`-pixel frame lying just inside `r`.
    pub fn outline(&mut self, r: Rect, thick: i32, color: (u8, u8, u8), alpha: f32) {
        let t = thick.min(r.w / 2).min(r.h / 2).max(1);
        self.fill(Rect::new(r.x, r.y, r.w, t), color, alpha);
        self.fill(Rect::new(r.x, r.bottom() - t, r.w, t), color, alpha);
        self.fill(Rect::new(r.x, r.y + t, t, r.h - 2 * t), color, alpha);
        self.fill(
            Rect::new(r.right() - t, r.y + t, t, r.h - 2 * t),
            color,
            alpha,
        );
    }

    /// After GDI drew text onto an opaque pill (GDI zeroes the alpha byte it touches), makes the
    /// pill opaque again and then fades it to `alpha`, keeping premultiplication right.
    pub fn restore_pill(&mut self, r: Rect, alpha: f32) {
        let Some((x0, y0, x1, y1)) = self.clip(r) else {
            return;
        };
        let a = (alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        for y in y0..y1 {
            let row = (y * self.w) as usize;
            for p in &mut self.px[row + x0 as usize..row + x1 as usize] {
                let ch = |shift: u32| ((((*p >> shift) & 0xFF) * a + 127) / 255) << shift;
                *p = a << 24 | ch(16) | ch(8) | ch(0);
            }
        }
    }
}

/// Where a label pill of the given text size goes: above the box, left-aligned with it; below it
/// when there is no room above; always kept inside `bounds` (the monitor).
pub fn label_rect(
    target: Rect,
    text_w: i32,
    text_h: i32,
    pad: i32,
    gap: i32,
    bounds: Rect,
) -> Rect {
    let (w, h) = (text_w + 2 * pad, text_h + pad);
    let above = target.y - gap - h;
    let y = if above >= bounds.y {
        above
    } else {
        target.bottom() + gap
    };
    let y = y.min(bounds.bottom() - h).max(bounds.y);
    let x = target.x.min(bounds.right() - w).max(bounds.x);
    Rect::new(x, y, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas(w: i32, h: i32) -> Vec<u32> {
        vec![0; (w * h) as usize]
    }

    fn is_orange(p: u32) -> bool {
        let (r, g, b, a) = unpack(p);
        a > 0 && r > g && g > b && (r as f32) > 1.5 * b as f32 + 10.0
    }

    #[test]
    fn premul_round_trips() {
        assert_eq!(premul(ORANGE, 1.0), 0xFFE8763A);
        assert_eq!(premul(ORANGE, 0.0), 0);
        let (r, _, _, a) = unpack(premul(ORANGE, 0.5));
        assert_eq!(a, 128);
        assert_eq!(r, 116);
    }

    #[test]
    fn over_blends_premultiplied() {
        let half_white = premul(WHITE, 0.5);
        assert_eq!(over(0, half_white), half_white);
        assert_eq!(over(half_white, 0), half_white);
        let (r, _, _, a) = unpack(over(premul(ORANGE, 1.0), half_white));
        assert_eq!(a, 255);
        assert!(r > 0xE8);
    }

    #[test]
    fn glow_is_orange_at_the_edges_and_clear_inside() {
        let (w, h) = (400, 300);
        let mut px = canvas(w, h);
        let mut c = Canvas::new(&mut px, w, h);
        c.glow(GLOW_PX, 1.0);
        for (x, y) in [
            (0, 150),
            (399, 150),
            (200, 0),
            (200, 299),
            (0, 0),
            (399, 299),
        ] {
            let p = c.get(x, y);
            assert!(is_orange(p), "({x},{y}) {:?}", unpack(p));
            assert!(unpack(p).3 >= 220, "edge should be strong: {:?}", unpack(p));
        }
        // Fades with distance from the edge, and the middle stays untouched (click-through look).
        let a = |x| unpack(c.get(x, 150)).3;
        assert!(a(0) > a(10) && a(10) > a(20) && a(20) > a(GLOW_PX - 1));
        assert_eq!(c.get(GLOW_PX, 150), 0);
        assert_eq!(c.get(200, 150), 0);
        assert_eq!(c.get(w - 1 - GLOW_PX, 150), 0);
    }

    #[test]
    fn glow_alpha_scales_and_zero_draws_nothing() {
        let (w, h) = (100, 100);
        let mut px = canvas(w, h);
        let mut c = Canvas::new(&mut px, w, h);
        c.glow(GLOW_PX, 0.0);
        assert!(c.px.iter().all(|&p| p == 0));
        c.glow(GLOW_PX, 0.5);
        let half = unpack(c.get(0, 50)).3;
        c.clear();
        c.glow(GLOW_PX, 1.0);
        let full = unpack(c.get(0, 50)).3;
        assert!(
            half < full && (half as i32 - full as i32 / 2).abs() <= 2,
            "{half} {full}"
        );
    }

    #[test]
    fn focus_box_draws_outline_fill_and_halo() {
        let (w, h) = (300, 200);
        let mut px = canvas(w, h);
        let mut c = Canvas::new(&mut px, w, h);
        let r = Rect::new(100, 80, 60, 30);
        c.focus_box(r, 1.0, 1.0);
        // Outline just outside the target, opaque.
        assert_eq!(unpack(c.get(99, 95)).3, 255);
        assert!(is_orange(c.get(99, 95)));
        assert_eq!(unpack(c.get(130, 77)).3, 255);
        // Faint fill inside, so what he must click stays readable.
        let inside = unpack(c.get(130, 95)).3;
        assert!(inside > 0 && inside < 40, "{inside}");
        // Halo fades outward, then nothing.
        let halo = unpack(c.get(130, 76)).3;
        assert!(halo > 0 && halo < 255);
        assert_eq!(c.get(130, 50), 0);
        assert_eq!(c.get(10, 10), 0);
    }

    #[test]
    fn drawing_clips_at_the_edges() {
        let (w, h) = (50, 50);
        let mut px = canvas(w, h);
        let mut c = Canvas::new(&mut px, w, h);
        c.focus_box(Rect::new(-20, -20, 40, 40), 2.0, 1.0);
        c.focus_box(Rect::new(1000, 1000, 40, 40), 1.0, 1.0);
        assert!(is_orange(c.get(21, 10)));
    }

    #[test]
    fn restore_pill_fixes_alpha_and_fades() {
        let (w, h) = (20, 10);
        let mut px = canvas(w, h);
        let mut c = Canvas::new(&mut px, w, h);
        let r = Rect::new(2, 2, 10, 5);
        c.fill(r, ORANGE, 1.0);
        // What GDI leaves behind after drawing white text: colour set, alpha zeroed.
        c.px[(3 * w + 4) as usize] = 0x00FFFFFF;
        c.restore_pill(r, 1.0);
        assert_eq!(c.get(4, 3), 0xFFFFFFFF);
        c.restore_pill(r, 0.5);
        let (r8, _, _, a) = unpack(c.get(4, 3));
        assert_eq!((r8, a), (128, 128));
        assert_eq!(c.get(0, 0), 0);
    }

    #[test]
    fn label_goes_above_or_below_and_stays_on_screen() {
        let mon = Rect::new(0, 0, 1920, 1080);
        let above = label_rect(Rect::new(100, 500, 200, 40), 80, 20, 6, 4, mon);
        assert_eq!(above, Rect::new(100, 500 - 4 - 26, 92, 26));
        let below = label_rect(Rect::new(100, 5, 200, 40), 80, 20, 6, 4, mon);
        assert_eq!(below.y, 49);
        let right = label_rect(Rect::new(1900, 500, 20, 20), 80, 20, 6, 4, mon);
        assert_eq!(right.right(), 1920);
        let second = label_rect(
            Rect::new(2000, 10, 50, 50),
            80,
            20,
            6,
            4,
            Rect::new(1920, 0, 1280, 1024),
        );
        assert!(second.x >= 1920 && second.y >= 0);
    }
}
