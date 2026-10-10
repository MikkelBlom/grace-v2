//! Audio I/O for the Grace engine: WASAPI capture, render and loopback on Windows.
//!
//! Talking to WASAPI means COM calls through the `windows` crate, which are `unsafe` by
//! construction. That unsafety is confined to this crate; everything it exposes is safe.
#![allow(unsafe_code)]

pub mod wasapi;
