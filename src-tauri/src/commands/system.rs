//! System-related Tauri commands.
//!
//! These commands provide app information and basic connectivity testing.

use serde::Serialize;

/// Application information returned by get_app_info.
#[derive(Debug, Clone, Serialize)]
pub struct AppInfo {
    /// Application name.
    pub name: String,
    /// Application version from Cargo.toml.
    pub version: String,
    /// Target platform (e.g., "macos", "windows", "linux").
    pub platform: String,
    /// Target architecture (e.g., "x86_64", "aarch64").
    pub arch: String,
    /// Whether running in debug mode.
    pub debug: bool,
}

/// Get application information.
///
/// Returns version, platform, and build configuration details.
#[tauri::command]
pub fn get_app_info() -> Result<AppInfo, String> {
    Ok(AppInfo {
        name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        debug: cfg!(debug_assertions),
    })
}

/// Simple ping command for connectivity testing.
///
/// Returns "pong" to verify IPC is working.
#[tauri::command]
pub fn ping() -> Result<String, String> {
    Ok("pong".to_string())
}

/// Echo command for connectivity testing.
///
/// Returns the input message unchanged.
#[tauri::command]
pub fn echo(message: String) -> Result<String, String> {
    Ok(message)
}
