//! Structured tracing for every Grace process.
//!
//! Every span and event is written as one JSON line to `<dir>/<session>.jsonl`. Spans are
//! written when they close, carrying their start time and duration, so a turn can be rebuilt
//! as a waterfall afterwards (see `grace-debug`).
//!
//! Writing happens on a dedicated thread behind a bounded channel. A full channel drops the
//! record and counts the drop instead of blocking: tracing must never stall the audio thread.
//!
//! A span field named `turn` tags the span and everything inside it with that turn id, so
//! per-turn latency falls out of the trace without any extra bookkeeping.

mod layer;
mod record;

pub use layer::{JsonlLayer, TraceGuard, init};
pub use record::{Kind, Record, read_session};
