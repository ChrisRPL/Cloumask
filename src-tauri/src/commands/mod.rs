//! Tauri IPC command handlers.

mod ollama;
mod sidecar;
mod system;

pub use ollama::*;
pub use sidecar::*;
pub use system::*;
