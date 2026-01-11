//! Cloumask Tauri application library.
//!
//! This is the main entry point for the Tauri application, handling:
//! - Python sidecar lifecycle (spawn on start, kill on close)
//! - IPC commands for frontend communication
//! - Application state management

mod commands;
mod sidecar;
mod state;

use sidecar::{SidecarConfig, SidecarManager};
use state::AppState;
use std::env;
use std::path::PathBuf;
use tauri::Manager;

#[tauri::command]
fn greet(name: &str) -> Result<String, String> {
    if name.trim().is_empty() {
        return Err("Name cannot be empty".to_string());
    }
    Ok(format!("Hello, {}! You've been greeted from Rust!", name))
}

/// Resolve the path to the backend/src directory for PYTHONPATH.
fn resolve_backend_src_path() -> String {
    // In development, navigate from current directory to backend/src
    if cfg!(debug_assertions) {
        // Try to find backend/src relative to the project root
        // When running with `cargo tauri dev`, CWD is the project root
        let candidates = [
            PathBuf::from("backend/src"),
            env::current_dir()
                .map(|p| p.join("backend/src"))
                .unwrap_or_default(),
        ];

        for candidate in &candidates {
            if candidate.exists() {
                log::info!("Found backend at: {:?}", candidate);
                return candidate.to_string_lossy().to_string();
            }
        }

        log::warn!("Backend path not found, using default");
        "backend/src".to_string()
    } else {
        // In production, the sidecar is bundled - PYTHONPATH not needed
        String::new()
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize logger for development
    #[cfg(debug_assertions)]
    {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            log::info!("Cloumask starting up...");

            // Configure sidecar with resolved backend path
            let config = SidecarConfig {
                backend_src_path: resolve_backend_src_path(),
                ..Default::default()
            };

            // Create sidecar manager
            let sidecar = SidecarManager::new(config);

            // Spawn sidecar process (non-blocking)
            if let Err(e) = sidecar.spawn() {
                log::error!("Failed to spawn sidecar: {}", e);
                // Don't fail app startup - sidecar can be restarted later
            }

            // Store state for use in commands
            let state = AppState::new(sidecar);
            let sidecar_ref = state.sidecar.clone();
            app.manage(state);

            // Wait for sidecar health in background (don't block window)
            std::thread::spawn(move || {
                if let Err(e) = sidecar_ref.wait_for_healthy() {
                    log::error!("Sidecar health check failed: {}", e);
                }
            });

            log::info!("Cloumask initialized successfully");
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                log::info!("Window close requested, shutting down sidecar...");

                // Kill sidecar when window is closed
                if let Some(state) = window.try_state::<AppState>() {
                    if let Err(e) = state.sidecar.kill() {
                        // NotRunning is fine, log other errors
                        if !matches!(e, sidecar::SidecarError::NotRunning) {
                            log::warn!("Failed to kill sidecar on close: {}", e);
                        }
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            commands::sidecar_status,
            commands::sidecar_base_url,
            commands::restart_sidecar,
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            eprintln!("Failed to start Cloumask: {e}");
            std::process::exit(1);
        });
}
