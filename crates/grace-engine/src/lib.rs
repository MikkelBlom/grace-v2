//! The `grace` engine as a library: the live binary (`main.rs`) and `grace-sim` share it, so the
//! simulator runs exactly the code Grace runs.

pub mod audio_io;
pub mod clock;
pub mod conductor;
pub mod delay;
pub mod frontend;
pub mod memory;
pub mod speech;
pub mod timed;
