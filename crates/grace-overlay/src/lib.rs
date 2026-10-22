//! Grace's screen overlay (docs/tools/vision.md): an orange edge glow while she works and focus
//! boxes with labels when she points at something.
//!
//! One layered window per monitor (WS_EX_LAYERED | TRANSPARENT | TOPMOST | NOACTIVATE |
//! TOOLWINDOW), painted with UpdateLayeredWindow from a CPU buffer. Grace v1 used Electron
//! transparent windows, which never reliably stayed visible on this dual-GPU laptop; a layered
//! window has no GPU or DirectComposition dependency.
//!
//! The drawing ([`draw`]), state ([`scene`]) and protocol ([`protocol`]) are pure and unit-tested;
//! `win` is the thin WinAPI layer.

pub mod draw;
pub mod protocol;
pub mod scene;
#[cfg(windows)]
pub mod win;
