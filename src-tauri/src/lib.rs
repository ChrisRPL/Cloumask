//! Cloumask Tauri application library.
//!
//! This is the main entry point for the Tauri application, handling:
//! - Python sidecar lifecycle (spawn on start, kill on close)
//! - IPC commands for frontend communication
//! - Application state management

mod commands;
mod docker;
mod pointcloud;
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
        // Try multiple strategies to find backend/src
        let mut candidates: Vec<PathBuf> = vec![
            // Direct path (if CWD is project root)
            PathBuf::from("backend/src"),
        ];

        // Try from current directory
        if let Ok(cwd) = env::current_dir() {
            candidates.push(cwd.join("backend/src"));
            // Try going up one level (if running from src-tauri/)
            candidates.push(cwd.join("../backend/src"));
        }

        // Try from CARGO_MANIFEST_DIR (compile-time path)
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        candidates.push(manifest_dir.join("../backend/src"));

        for candidate in &candidates {
            // Canonicalize to resolve .. and check existence
            if let Ok(abs_path) = candidate.canonicalize() {
                log::info!("Found backend at: {:?}", abs_path);
                return abs_path.to_string_lossy().to_string();
            }
        }

        log::warn!("Backend path not found in any candidate location");
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
        .plugin(tauri_plugin_dialog::init())
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
            // Sidecar lifecycle commands
            commands::sidecar_status,
            commands::start_sidecar,
            commands::stop_sidecar,
            commands::restart_sidecar,
            // Health check commands
            commands::check_health,
            commands::check_ready,
            // Generic sidecar HTTP commands
            commands::call_sidecar_get,
            commands::call_sidecar_post,
            // LLM commands
            commands::get_llm_status,
            commands::list_llm_models,
            // System commands
            commands::get_app_info,
            commands::ping,
            commands::echo,
            // Point cloud commands
            commands::read_pointcloud_metadata,
            commands::read_pointcloud,
            commands::stream_pointcloud,
            commands::convert_pointcloud,
            commands::decimate_pointcloud,
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            eprintln!("Failed to start Cloumask: {e}");
            std::process::exit(1);
        });
}
