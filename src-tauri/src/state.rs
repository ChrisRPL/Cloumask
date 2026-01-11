//! Application state management.
//!
//! Holds shared state accessible across Tauri commands.

use std::sync::Arc;

use crate::sidecar::SidecarManager;

/// Global application state shared across Tauri commands.
pub struct AppState {
    /// Manager for the Python sidecar process.
    pub sidecar: Arc<SidecarManager>,
}

impl AppState {
    /// Create a new app state with a sidecar manager.
    pub fn new(sidecar: SidecarManager) -> Self {
        Self {
            sidecar: Arc::new(sidecar),
        }
    }
}
