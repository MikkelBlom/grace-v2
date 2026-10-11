//! The engine's one clock: the Windows performance counter in seconds, the same clock WASAPI
//! stamps audio packets with, so "now" and "when this sample was captured" are comparable.
#![allow(unsafe_code)]

use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

pub fn now() -> f64 {
    let (mut count, mut freq) = (0i64, 0i64);
    // Both calls cannot fail on Windows XP and later.
    unsafe {
        let _ = QueryPerformanceCounter(&mut count);
        let _ = QueryPerformanceFrequency(&mut freq);
    }
    count as f64 / freq as f64
}
