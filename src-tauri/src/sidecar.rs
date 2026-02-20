//! Python sidecar process management and HTTP communication.
//!
//! Handles spawning, monitoring, and terminating the Python FastAPI backend.
//! The sidecar runs uvicorn serving `backend.api.main:app` on port 8765.
//!
//! Provides both blocking and async HTTP methods for communicating with the sidecar.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
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

    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("Failed to deserialize response: {0}")]
    DeserializationError(String),

    #[error("Sidecar returned error status: {status} - {message}")]
    SidecarResponseError { status: u16, message: String },
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
    /// Maximum number of retry attempts for failed requests.
    #[allow(dead_code)]
    pub retry_attempts: u32,
    /// Initial delay between retries in milliseconds.
    #[allow(dead_code)]
    pub retry_delay_ms: u64,
    /// Maximum delay between retries in milliseconds (for exponential backoff cap).
    #[allow(dead_code)]
    pub retry_max_delay_ms: u64,
    /// Request timeout in seconds.
    pub request_timeout_secs: u64,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8765,
            python_path: "python3".to_string(),
            backend_src_path: "backend/src".to_string(),
            health_check_timeout_secs: 30,
            retry_attempts: 3,
            retry_delay_ms: 100,
            retry_max_delay_ms: 2000,
            request_timeout_secs: 30,
        }
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Health check response from Python sidecar.
///
/// Returned by the `/health` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Current health status: "healthy", "degraded", or "unhealthy".
    pub status: String,
    /// Backend version.
    pub version: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Status of individual components.
    pub components: HashMap<String, String>,
    /// Absolute backend/src path reported by the sidecar.
    #[serde(default)]
    pub backend_src_path: Option<String>,
}

/// Readiness check response from Python sidecar.
///
/// Returned by the `/ready` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyResponse {
    /// Whether the sidecar is ready to accept requests.
    pub ready: bool,
    /// Individual readiness checks.
    pub checks: HashMap<String, bool>,
    /// Absolute backend/src path reported by the sidecar.
    #[serde(default)]
    pub backend_src_path: Option<String>,
}

// ============================================================================
// Sidecar Manager
// ============================================================================

/// Manages the Python sidecar process lifecycle and HTTP communication.
pub struct SidecarManager {
    /// The running child process, if any.
    process: Mutex<Option<Child>>,
    /// Configuration for the sidecar.
    config: SidecarConfig,
    /// Async HTTP client for sidecar communication.
    client: Client,
}

impl SidecarManager {
    fn normalize_path(path: &str) -> String {
        Path::new(path)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(path))
            .to_string_lossy()
            .to_string()
    }

    fn expected_backend_src_path(&self) -> String {
        Self::normalize_path(&self.config.backend_src_path)
    }

    fn backend_src_path_matches(&self, reported_path: Option<&str>) -> bool {
        let Some(path) = reported_path else {
            return false;
        };
        Self::normalize_path(path) == self.expected_backend_src_path()
    }

    fn unexpected_backend_error(
        &self,
        endpoint: &str,
        reported_path: Option<&str>,
    ) -> SidecarError {
        SidecarError::HttpError(format!(
            "Unexpected sidecar instance at {} (expected backend/src '{}', got '{}')",
            endpoint,
            self.expected_backend_src_path(),
            reported_path.unwrap_or("<missing>")
        ))
    }

    /// Create a new sidecar manager with the given configuration.
    pub fn new(config: SidecarConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            // Avoid macOS system proxy resolution panics in sandboxed/test envs.
            .no_proxy()
            .build()
            .expect("Failed to create HTTP client");

        Self {
            process: Mutex::new(None),
            config,
            client,
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

        // Build the uvicorn command.
        // Use backend workdir to ensure env-file resolution and imports are tied
        // to this repository, not whatever cwd/path the parent process had.
        let mut command = Command::new(&self.config.python_path);
        command
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
            // Enable lightweight 3D detection fallback when OpenPCDet weights are missing.
            .env("CLOUMASK_ENABLE_3D_HEURISTIC_FALLBACK", "1")
            // Avoid pipe backpressure deadlocks when no reader is attached.
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        if let Some(workdir) = Path::new(&self.config.backend_src_path).parent() {
            command.current_dir(workdir);
        }

        let child = command.spawn()?;

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
            // Match async client behavior and avoid platform proxy lookup side effects.
            .no_proxy()
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
                            let reported_path = json.get("backend_src_path").and_then(|v| v.as_str());
                            if self.backend_src_path_matches(reported_path) {
                                log::info!("Sidecar is healthy and ready");
                                return Ok(());
                            }
                            log::warn!(
                                "{}",
                                self.unexpected_backend_error("/health", reported_path)
                            );
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

    // ========================================================================
    // Async HTTP Methods
    // ========================================================================

    /// Perform an async GET request to the sidecar.
    ///
    /// Returns the deserialized response of type `T`.
    pub async fn get_async<T>(&self, endpoint: &str) -> Result<T, SidecarError>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = format!("{}{}", self.base_url(), endpoint);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| SidecarError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(SidecarError::SidecarResponseError { status, message });
        }

        response
            .json::<T>()
            .await
            .map_err(|e| SidecarError::DeserializationError(e.to_string()))
    }

    /// Perform an async POST request to the sidecar with JSON body.
    ///
    /// Returns the deserialized response of type `T`.
    pub async fn post_async<T, B>(&self, endpoint: &str, body: &B) -> Result<T, SidecarError>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {
        let url = format!("{}{}", self.base_url(), endpoint);

        let response = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| SidecarError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(SidecarError::SidecarResponseError { status, message });
        }

        response
            .json::<T>()
            .await
            .map_err(|e| SidecarError::DeserializationError(e.to_string()))
    }

    /// Perform a request with exponential backoff retry.
    ///
    /// The `operation` closure is called repeatedly until it succeeds or
    /// the maximum number of retries is reached.
    #[allow(dead_code)]
    pub async fn request_with_retry<T, F, Fut>(&self, operation: F) -> Result<T, SidecarError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, SidecarError>>,
    {
        let mut delay = Duration::from_millis(self.config.retry_delay_ms);
        let max_delay = Duration::from_millis(self.config.retry_max_delay_ms);
        let mut last_error = None;

        for attempt in 0..=self.config.retry_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    log::debug!("Request attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);

                    if attempt < self.config.retry_attempts {
                        tokio::time::sleep(delay).await;
                        delay = std::cmp::min(delay * 2, max_delay);
                    }
                }
            }
        }

        Err(last_error.unwrap_or(SidecarError::HttpError(
            "Unknown error after retries".to_string(),
        )))
    }

    /// Async health check - calls /health endpoint.
    ///
    /// Returns the full health response from the sidecar.
    pub async fn health_check_async(&self) -> Result<HealthResponse, SidecarError> {
        let response = self.get_async::<HealthResponse>("/health").await?;
        if self.backend_src_path_matches(response.backend_src_path.as_deref()) {
            Ok(response)
        } else {
            Err(self.unexpected_backend_error(
                "/health",
                response.backend_src_path.as_deref(),
            ))
        }
    }

    /// Async readiness check - calls /ready endpoint.
    ///
    /// Returns the readiness response from the sidecar.
    pub async fn ready_check_async(&self) -> Result<ReadyResponse, SidecarError> {
        let response = self.get_async::<ReadyResponse>("/ready").await?;
        if self.backend_src_path_matches(response.backend_src_path.as_deref()) {
            Ok(response)
        } else {
            Err(self.unexpected_backend_error(
                "/ready",
                response.backend_src_path.as_deref(),
            ))
        }
    }

    /// Wait for the sidecar to become ready by polling /health endpoint (async version).
    ///
    /// This is the async version of `wait_for_healthy()`.
    #[allow(dead_code)]
    pub async fn wait_for_ready_async(&self) -> Result<HealthResponse, SidecarError> {
        let start = Instant::now();
        let timeout = Duration::from_secs(self.config.health_check_timeout_secs);
        let check_interval = Duration::from_millis(500);

        log::info!("Waiting for sidecar to become healthy (async)...");

        loop {
            if start.elapsed() > timeout {
                return Err(SidecarError::HealthCheckTimeout(
                    self.config.health_check_timeout_secs,
                ));
            }

            // Check if process is still running
            if !self.is_running() {
                return Err(SidecarError::NotRunning);
            }

            match self.health_check_async().await {
                Ok(health) if health.status == "healthy" => {
                    log::info!("Sidecar is healthy and ready (async)");
                    return Ok(health);
                }
                Ok(health) => {
                    log::debug!("Sidecar status: {} (waiting for healthy)", health.status);
                }
                Err(e) => {
                    log::debug!("Health check failed (retrying): {}", e);
                }
            }

            tokio::time::sleep(check_interval).await;
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

    #[test]
    fn test_config_with_retry_defaults() {
        let config = SidecarConfig::default();
        assert_eq!(config.retry_attempts, 3);
        assert_eq!(config.retry_delay_ms, 100);
        assert_eq!(config.retry_max_delay_ms, 2000);
        assert_eq!(config.request_timeout_secs, 30);
    }

    #[test]
    fn test_health_response_deserialize() {
        let json = r#"{
            "status": "healthy",
            "version": "0.1.0",
            "timestamp": "2025-01-11T12:00:00Z",
            "components": {"api": "healthy", "agent": "not_loaded"}
        }"#;

        let response: HealthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "healthy");
        assert_eq!(response.version, "0.1.0");
        assert_eq!(
            response.components.get("api"),
            Some(&"healthy".to_string())
        );
        assert_eq!(
            response.components.get("agent"),
            Some(&"not_loaded".to_string())
        );
        assert_eq!(response.backend_src_path, None);
    }

    #[test]
    fn test_ready_response_deserialize() {
        let json = r#"{
            "ready": true,
            "checks": {"api_running": true, "routes_loaded": true}
        }"#;

        let response: ReadyResponse = serde_json::from_str(json).unwrap();
        assert!(response.ready);
        assert_eq!(response.checks.get("api_running"), Some(&true));
        assert_eq!(response.checks.get("routes_loaded"), Some(&true));
        assert_eq!(response.backend_src_path, None);
    }

    #[test]
    fn test_health_response_serialize() {
        let mut components = HashMap::new();
        components.insert("api".to_string(), "healthy".to_string());

        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
            timestamp: "2025-01-11T12:00:00Z".to_string(),
            components,
            backend_src_path: Some("/tmp/backend/src".to_string()),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"version\":\"0.1.0\""));
        assert!(json.contains("\"backend_src_path\":\"/tmp/backend/src\""));
    }

    #[test]
    fn test_backend_src_path_matching() {
        let manager = SidecarManager::with_defaults();
        let expected = SidecarManager::normalize_path(&manager.config.backend_src_path);
        assert!(manager.backend_src_path_matches(Some(expected.as_str())));
        assert!(!manager.backend_src_path_matches(Some("/tmp/other-backend/src")));
        assert!(!manager.backend_src_path_matches(None));
    }
}
