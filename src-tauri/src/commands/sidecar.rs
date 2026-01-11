//! Sidecar-related Tauri commands.
//!
//! These commands allow the frontend to query and control the Python sidecar.

use tauri::State;

use crate::state::AppState;

/// Check if the sidecar process is currently running.
#[tauri::command]
pub fn sidecar_status(state: State<'_, AppState>) -> Result<bool, String> {
    Ok(state.sidecar.is_running())
}

/// Get the base URL for the sidecar API.
#[tauri::command]
pub fn sidecar_base_url(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.sidecar.base_url())
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
