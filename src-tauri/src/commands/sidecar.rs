//! Sidecar-related Tauri commands.
//!
//! These commands allow the frontend to query and control the Python sidecar.

use serde::Serialize;
use tauri::State;

use crate::sidecar::SidecarError;
use crate::state::AppState;

/// Status information for the Python sidecar process.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarStatus {
    /// Whether the sidecar process is currently running.
    pub running: bool,
    /// The base URL for the sidecar API (e.g., "http://127.0.0.1:8765").
    pub url: String,
    /// The port the sidecar is configured to use.
    pub port: u16,
}

/// Get the current status of the sidecar process.
///
/// Returns comprehensive status including running state, URL, and port.
#[tauri::command]
pub fn sidecar_status(state: State<'_, AppState>) -> Result<SidecarStatus, String> {
    Ok(SidecarStatus {
        running: state.sidecar.is_running(),
        url: state.sidecar.base_url(),
        port: state.sidecar.port(),
    })
}

/// Start the sidecar process if not already running.
///
/// Spawns the Python FastAPI backend. If already running, returns success without action.
/// The spawn is non-blocking - use `sidecar_status` to poll for readiness.
#[tauri::command]
pub fn start_sidecar(state: State<'_, AppState>) -> Result<(), String> {
    if state.sidecar.is_running() {
        log::info!("Sidecar already running, skipping start");
        return Ok(());
    }

    state.sidecar.spawn().map_err(|e| e.to_string())?;

    // Start health check in background
    let sidecar = state.sidecar.clone();
    std::thread::spawn(move || {
        if let Err(e) = sidecar.wait_for_healthy() {
            log::error!("Sidecar health check failed after start: {}", e);
        }
    });

    Ok(())
}

/// Stop the sidecar process.
///
/// Gracefully terminates the Python backend. Returns success if already stopped.
#[tauri::command]
pub fn stop_sidecar(state: State<'_, AppState>) -> Result<(), String> {
    match state.sidecar.kill() {
        Ok(()) => Ok(()),
        Err(SidecarError::NotRunning) => {
            log::info!("Sidecar already stopped");
            Ok(())
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Restart the sidecar process.
///
/// Kills the existing sidecar (if running) and spawns a new one.
/// The spawn is non-blocking - use `sidecar_status` to poll for readiness.
#[tauri::command]
pub fn restart_sidecar(state: State<'_, AppState>) -> Result<(), String> {
    // Kill if running (ignore NotRunning error)
    let _ = state.sidecar.kill();

    // Respawn (non-blocking) - frontend should poll sidecar_status
    state.sidecar.spawn().map_err(|e| e.to_string())?;

    // Start health check in background
    let sidecar = state.sidecar.clone();
    std::thread::spawn(move || {
        if let Err(e) = sidecar.wait_for_healthy() {
            log::error!("Sidecar health check failed after restart: {}", e);
        }
    });

    Ok(())
}
