//! Frame cost of the glow on a 2560x1600 canvas (cargo run --release --example paint_cost).
use grace_overlay::draw::{Canvas, GLOW_PX};
fn main() {
    let (w, h) = (2560, 1600);
    let mut px = vec![0u32; (w * h) as usize];
    let t = std::time::Instant::now();
    for i in 0..100 {
        Canvas::new(&mut px, w, h).glow((GLOW_PX as f32 * 1.5) as i32, 0.5 + (i as f32) / 400.0);
        std::hint::black_box(&px);
    }
    println!("glow frame: {:.2} ms", t.elapsed().as_secs_f64() * 10.0);
}
