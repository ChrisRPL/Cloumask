//! Sidecar-related Tauri commands.
//!
//! These commands allow the frontend to query and control the Python sidecar,
//! including health checks and generic HTTP requests to sidecar endpoints.

use serde::Serialize;
use tauri::State;

use crate::sidecar::SidecarError;
use crate::state::AppState;

// Re-export response types for use in other modules
pub use crate::sidecar::{HealthResponse, ReadyResponse};

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

// ============================================================================
// Health Check Commands
// ============================================================================

/// Check the health of the Python sidecar.
///
/// Calls the /health endpoint and returns the full health response.
#[tauri::command]
pub async fn check_health(state: State<'_, AppState>) -> Result<HealthResponse, String> {
    state
        .sidecar
        .health_check_async()
        .await
        .map_err(|e| e.to_string())
}

/// Check the readiness of the Python sidecar.
///
/// Calls the /ready endpoint and returns the readiness response.
#[tauri::command]
pub async fn check_ready(state: State<'_, AppState>) -> Result<ReadyResponse, String> {
    state
        .sidecar
        .ready_check_async()
        .await
        .map_err(|e| e.to_string())
}

// ============================================================================
// Generic Sidecar HTTP Commands (for debugging/development)
// ============================================================================

/// Generic GET request to the sidecar.
///
/// Allows calling any sidecar endpoint from the frontend.
/// Returns the raw JSON response as a Value.
#[tauri::command]
pub async fn call_sidecar_get(
    state: State<'_, AppState>,
    endpoint: String,
) -> Result<serde_json::Value, String> {
    // Validate endpoint starts with /
    let endpoint = if endpoint.starts_with('/') {
        endpoint
    } else {
        format!("/{}", endpoint)
    };

    state
        .sidecar
        .get_async::<serde_json::Value>(&endpoint)
        .await
        .map_err(|e| e.to_string())
}

/// Generic POST request to the sidecar.
///
/// Allows calling any sidecar endpoint with a JSON body.
/// Returns the raw JSON response as a Value.
#[tauri::command]
pub async fn call_sidecar_post(
    state: State<'_, AppState>,
    endpoint: String,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let endpoint = if endpoint.starts_with('/') {
        endpoint
    } else {
        format!("/{}", endpoint)
    };

    state
        .sidecar
        .post_async::<serde_json::Value, serde_json::Value>(&endpoint, &body)
        .await
        .map_err(|e| e.to_string())
}
