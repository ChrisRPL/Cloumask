//! Python sidecar process management.
//!
//! Handles spawning, monitoring, and terminating the Python FastAPI backend.
//! The sidecar runs uvicorn serving `backend.api.main:app` on port 8765.

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use thiserror::Error;

/// Errors that can occur during sidecar operations.
#[derive(Error, Debug)]
pub enum SidecarError {
    #[error("Failed to spawn sidecar process: {0}")]
    SpawnError(#[from] std::io::Error),

    #[error("Sidecar process not running")]
    NotRunning,

    #[error("Failed to acquire sidecar lock")]
    LockError,

    #[error("Sidecar health check failed after {0} seconds")]
    HealthCheckTimeout(u64),
}

/// Configuration for the sidecar process.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// Host to bind the sidecar server.
    pub host: String,
    /// Port for the sidecar server.
    pub port: u16,
    /// Path to Python executable.
    pub python_path: String,
    /// Path to the backend/src directory (for PYTHONPATH).
    pub backend_src_path: String,
    /// Timeout for health check in seconds.
    pub health_check_timeout_secs: u64,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8765,
            python_path: "python3".to_string(),
            backend_src_path: "backend/src".to_string(),
            health_check_timeout_secs: 30,
        }
    }
}

/// Manages the Python sidecar process lifecycle.
pub struct SidecarManager {
    /// The running child process, if any.
    process: Mutex<Option<Child>>,
    /// Configuration for the sidecar.
    config: SidecarConfig,
}

impl SidecarManager {
    /// Create a new sidecar manager with the given configuration.
    pub fn new(config: SidecarConfig) -> Self {
        Self {
            process: Mutex::new(None),
            config,
        }
    }

    /// Create a new sidecar manager with default configuration.
    #[cfg(test)]
    pub fn with_defaults() -> Self {
        Self::new(SidecarConfig::default())
    }

    /// Spawn the Python sidecar process.
    ///
    /// Starts uvicorn with the FastAPI backend. If a process is already running,
    /// it will be killed first.
    pub fn spawn(&self) -> Result<(), SidecarError> {
        let mut process_guard = self.process.lock().map_err(|_| SidecarError::LockError)?;

        // Kill existing process if running
        if let Some(mut existing) = process_guard.take() {
            log::info!("Killing existing sidecar before respawn");
            let _ = existing.kill();
            let _ = existing.wait();
        }

        // Build the uvicorn command
        let child = Command::new(&self.config.python_path)
            .args([
                "-m",
                "uvicorn",
                "backend.api.main:app",
                "--host",
                &self.config.host,
                "--port",
                &self.config.port.to_string(),
                "--log-level",
                "info",
            ])
            .env("PYTHONPATH", &self.config.backend_src_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        log::info!("Sidecar spawned with PID: {}", child.id());

        *process_guard = Some(child);
        Ok(())
    }

    /// Kill the sidecar process gracefully.
    ///
    /// On Unix, sends SIGTERM first for graceful shutdown, then SIGKILL if needed.
    /// On Windows, terminates the process directly.
    pub fn kill(&self) -> Result<(), SidecarError> {
        let mut process_guard = self.process.lock().map_err(|_| SidecarError::LockError)?;

        if let Some(mut child) = process_guard.take() {
            let pid = child.id();
            log::info!("Killing sidecar process (PID: {})...", pid);

            // On Unix, try graceful shutdown with SIGTERM first
            #[cfg(unix)]
            {
                // Send SIGTERM for graceful shutdown
                unsafe {
                    libc::kill(pid as i32, libc::SIGTERM);
                }

                // Wait a bit for graceful shutdown
                std::thread::sleep(Duration::from_secs(2));

                // Check if process exited
                match child.try_wait() {
                    Ok(Some(status)) => {
                        log::info!("Sidecar exited gracefully with status: {:?}", status);
                        return Ok(());
                    }
                    Ok(None) => {
                        log::warn!("Sidecar still running after SIGTERM, forcing termination...");
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    Err(e) => {
                        log::error!("Error checking sidecar status: {}", e);
                        let _ = child.kill();
                    }
                }
            }

            // On Windows (or as fallback), just kill directly
            #[cfg(not(unix))]
            {
                let _ = child.kill();
                let _ = child.wait();
            }

            log::info!("Sidecar killed successfully");
            Ok(())
        } else {
            Err(SidecarError::NotRunning)
        }
    }

    /// Check if the sidecar process is running.
    pub fn is_running(&self) -> bool {
        let Ok(mut process_guard) = self.process.lock() else {
            return false;
        };

        if let Some(ref mut child) = *process_guard {
            match child.try_wait() {
                Ok(Some(_)) => {
                    // Process has exited
                    *process_guard = None;
                    false
                }
                Ok(None) => true, // Still running
                Err(_) => false,
            }
        } else {
            false
        }
    }

    /// Get the sidecar's base URL.
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.config.host, self.config.port)
    }

    /// Get the sidecar's configured port.
    pub fn port(&self) -> u16 {
        self.config.port
    }

    /// Wait for the sidecar to become healthy by polling the /health endpoint.
    pub fn wait_for_healthy(&self) -> Result<(), SidecarError> {
        let start = Instant::now();
        let timeout = Duration::from_secs(self.config.health_check_timeout_secs);
        let check_interval = Duration::from_millis(500);
        let health_url = format!("{}/health", self.base_url());

        log::info!("Waiting for sidecar to become healthy at {}...", health_url);

        // Use blocking reqwest client since we're in sync context
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| SidecarError::SpawnError(std::io::Error::other(e.to_string())))?;

        loop {
            if start.elapsed() > timeout {
                return Err(SidecarError::HealthCheckTimeout(
                    self.config.health_check_timeout_secs,
                ));
            }

            // Check if process is still running
            if !self.is_running() {
                return Err(SidecarError::SpawnError(std::io::Error::other(
                    "Sidecar process exited unexpectedly",
                )));
            }

            match client.get(&health_url).send() {
                Ok(response) if response.status().is_success() => {
                    if let Ok(json) = response.json::<serde_json::Value>() {
                        if json.get("status") == Some(&serde_json::json!("healthy")) {
                            log::info!("Sidecar is healthy and ready");
                            return Ok(());
                        }
                    }
                }
                Ok(response) => {
                    log::debug!(
                        "Health check returned non-success status: {}",
                        response.status()
                    );
                }
                Err(e) => {
                    log::debug!("Health check failed (retrying): {}", e);
                }
            }

            std::thread::sleep(check_interval);
        }
    }
}

impl Drop for SidecarManager {
    fn drop(&mut self) {
        // Ensure sidecar is killed when manager is dropped
        if let Err(e) = self.kill() {
            // NotRunning is fine, any other error should be logged
            if !matches!(e, SidecarError::NotRunning) {
                log::warn!("Failed to kill sidecar on drop: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SidecarConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8765);
        assert_eq!(config.python_path, "python3");
    }

    #[test]
    fn test_base_url() {
        let manager = SidecarManager::with_defaults();
        assert_eq!(manager.base_url(), "http://127.0.0.1:8765");
    }

    #[test]
    fn test_port() {
        let manager = SidecarManager::with_defaults();
        assert_eq!(manager.port(), 8765);
    }

    #[test]
    fn test_not_running_initially() {
        let manager = SidecarManager::with_defaults();
        assert!(!manager.is_running());
    }
}
