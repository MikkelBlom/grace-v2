//! Pure, IO-free logic for Grace. Everything here is deterministic and driven by explicit
//! inputs and a caller-supplied clock, so it is tested without audio, models or threads.

pub mod turn;
