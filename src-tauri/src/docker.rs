//! Docker container management for Cloumask.
//!
//! Manages the LLM container lifecycle using bollard for direct socket communication.
//! Supports Docker Engine, Colima, and Podman socket auto-detection.

use std::collections::HashMap;
use std::time::Duration;

use bollard::container::{
    Config, CreateContainerOptions, ListContainersOptions, RemoveContainerOptions,
    StartContainerOptions, StopContainerOptions,
};
use bollard::models::{ContainerSummary, HostConfig, PortBinding};
use bollard::Docker;
use thiserror::Error;

/// Container name for the LLM service.
const LLM_CONTAINER_NAME: &str = "cloumask-llm";
/// Image name for the LLM service.
const LLM_IMAGE_NAME: &str = "cloumask-llm:latest";
/// Port for the LLM service.
const LLM_PORT: u16 = 11434;

/// Errors that can occur during Docker operations.
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum DockerError {
    #[error("Docker daemon not available: {0}")]
    DaemonNotAvailable(String),

    #[error("Container operation failed: {0}")]
    ContainerError(String),

    #[error("Image operation failed: {0}")]
    ImageError(String),

    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),

    #[error("Timeout waiting for container: {0}")]
    Timeout(String),

    #[error("Bollard error: {0}")]
    Bollard(#[from] bollard::errors::Error),
}

/// Status of the Docker daemon.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DockerStatus {
    /// Whether Docker daemon is available.
    pub available: bool,
    /// Docker socket path being used.
    pub socket_path: Option<String>,
    /// Error message if unavailable.
    pub error: Option<String>,
}

/// Status of the LLM container.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ContainerStatus {
    /// Whether the container exists.
    pub exists: bool,
    /// Whether the container is running.
    pub running: bool,
    /// Container health status.
    pub healthy: bool,
    /// Container ID if exists.
    pub container_id: Option<String>,
}

/// Manages Docker containers for Cloumask.
///
/// Note: Methods are scaffolded for future integration (Phase 4/5).
/// Currently not called from app lifecycle - will be integrated when
/// Docker-based execution is enabled.
#[allow(dead_code)]
pub struct DockerManager {
    client: Docker,
}

#[allow(dead_code)]
impl DockerManager {
    /// Create a new DockerManager, auto-detecting the Docker socket.
    ///
    /// Tries multiple socket paths in order:
    /// 1. Default Docker socket (/var/run/docker.sock on Unix)
    /// 2. Colima socket (~/.colima/default/docker.sock)
    /// 3. Podman socket
    pub async fn new() -> Result<Self, DockerError> {
        // Try default Docker socket first
        if let Ok(client) = Docker::connect_with_socket_defaults() {
            // Verify connection works
            if client.ping().await.is_ok() {
                log::info!("Connected to Docker via default socket");
                return Ok(Self { client });
            }
        }

        // Try Colima socket on macOS
        #[cfg(target_os = "macos")]
        {
            let home = std::env::var("HOME").unwrap_or_default();
            let colima_socket = format!("{}/.colima/default/docker.sock", home);
            if std::path::Path::new(&colima_socket).exists() {
                if let Ok(client) = Docker::connect_with_socket(
                    &colima_socket,
                    120,
                    bollard::API_DEFAULT_VERSION,
                ) {
                    if client.ping().await.is_ok() {
                        log::info!("Connected to Docker via Colima socket");
                        return Ok(Self { client });
                    }
                }
            }
        }

        // Try local defaults as fallback
        let client = Docker::connect_with_local_defaults()
            .map_err(|e| DockerError::DaemonNotAvailable(e.to_string()))?;

        // Verify connection
        client
            .ping()
            .await
            .map_err(|e| DockerError::DaemonNotAvailable(e.to_string()))?;

        log::info!("Connected to Docker via local defaults");
        Ok(Self { client })
    }

    /// Check if Docker daemon is available.
    pub async fn check_status(&self) -> DockerStatus {
        match self.client.ping().await {
            Ok(_) => DockerStatus {
                available: true,
                socket_path: None, // Would need to track this
                error: None,
            },
            Err(e) => DockerStatus {
                available: false,
                socket_path: None,
                error: Some(e.to_string()),
            },
        }
    }

    /// Get the status of the LLM container.
    pub async fn get_llm_container_status(&self) -> Result<ContainerStatus, DockerError> {
        let filters: HashMap<String, Vec<String>> = HashMap::from([(
            "name".to_string(),
            vec![LLM_CONTAINER_NAME.to_string()],
        )]);

        let options = ListContainersOptions {
            all: true,
            filters,
            ..Default::default()
        };

        let containers: Vec<ContainerSummary> = self.client.list_containers(Some(options)).await?;

        if let Some(container) = containers.first() {
            let running = container
                .state
                .as_ref()
                .map(|s| s == "running")
                .unwrap_or(false);

            // Check health if running
            let healthy = if running {
                self.check_llm_health().await.unwrap_or(false)
            } else {
                false
            };

            Ok(ContainerStatus {
                exists: true,
                running,
                healthy,
                container_id: container.id.clone(),
            })
        } else {
            Ok(ContainerStatus {
                exists: false,
                running: false,
                healthy: false,
                container_id: None,
            })
        }
    }

    /// Check if the LLM service is healthy by pinging its API.
    async fn check_llm_health(&self) -> Result<bool, DockerError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| DockerError::HealthCheckFailed(e.to_string()))?;

        match client
            .get(format!("http://localhost:{}/api/tags", LLM_PORT))
            .send()
            .await
        {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Start the LLM container.
    ///
    /// If the container doesn't exist, creates it first.
    /// If the container is stopped, starts it.
    pub async fn start_llm_container(&self) -> Result<(), DockerError> {
        let status = self.get_llm_container_status().await?;

        if status.running {
            log::info!("LLM container is already running");
            return Ok(());
        }

        if status.exists {
            // Container exists but is stopped, start it
            log::info!("Starting existing LLM container");
            self.client
                .start_container(LLM_CONTAINER_NAME, None::<StartContainerOptions<String>>)
                .await?;
        } else {
            // Create and start new container
            log::info!("Creating new LLM container");
            self.create_llm_container().await?;
        }

        Ok(())
    }

    /// Create a new LLM container.
    async fn create_llm_container(&self) -> Result<(), DockerError> {
        // Port bindings
        let mut port_bindings: HashMap<String, Option<Vec<PortBinding>>> = HashMap::new();
        port_bindings.insert(
            format!("{}/tcp", LLM_PORT),
            Some(vec![PortBinding {
                host_ip: Some("127.0.0.1".to_string()),
                host_port: Some(LLM_PORT.to_string()),
            }]),
        );

        let host_config = HostConfig {
            port_bindings: Some(port_bindings),
            // Memory limit: 16GB
            memory: Some(16 * 1024 * 1024 * 1024),
            // Restart unless stopped
            restart_policy: Some(bollard::models::RestartPolicy {
                name: Some(bollard::models::RestartPolicyNameEnum::UNLESS_STOPPED),
                maximum_retry_count: None,
            }),
            ..Default::default()
        };

        let config = Config {
            image: Some(LLM_IMAGE_NAME.to_string()),
            host_config: Some(host_config),
            ..Default::default()
        };

        let options = CreateContainerOptions {
            name: LLM_CONTAINER_NAME,
            platform: None,
        };

        self.client.create_container(Some(options), config).await?;

        // Start the container
        self.client
            .start_container(LLM_CONTAINER_NAME, None::<StartContainerOptions<String>>)
            .await?;

        Ok(())
    }

    /// Stop the LLM container.
    pub async fn stop_llm_container(&self) -> Result<(), DockerError> {
        let status = self.get_llm_container_status().await?;

        if !status.running {
            log::info!("LLM container is not running");
            return Ok(());
        }

        log::info!("Stopping LLM container");
        let options = StopContainerOptions { t: 10 };
        self.client
            .stop_container(LLM_CONTAINER_NAME, Some(options))
            .await?;

        Ok(())
    }

    /// Remove the LLM container.
    pub async fn remove_llm_container(&self) -> Result<(), DockerError> {
        // Stop first if running
        self.stop_llm_container().await.ok();

        let status = self.get_llm_container_status().await?;
        if !status.exists {
            return Ok(());
        }

        log::info!("Removing LLM container");
        let options = RemoveContainerOptions {
            force: true,
            ..Default::default()
        };
        self.client
            .remove_container(LLM_CONTAINER_NAME, Some(options))
            .await?;

        Ok(())
    }

    /// Wait for the LLM container to become healthy.
    pub async fn wait_for_healthy(&self, timeout: Duration) -> Result<(), DockerError> {
        let start = std::time::Instant::now();
        let poll_interval = Duration::from_secs(2);

        log::info!("Waiting for LLM container to become healthy...");

        while start.elapsed() < timeout {
            if self.check_llm_health().await.unwrap_or(false) {
                log::info!("LLM container is healthy");
                return Ok(());
            }
            tokio::time::sleep(poll_interval).await;
        }

        Err(DockerError::Timeout(format!(
            "LLM container did not become healthy within {:?}",
            timeout
        )))
    }

    /// Pull the LLM image if it doesn't exist.
    pub async fn ensure_llm_image(&self) -> Result<(), DockerError> {
        // Check if image exists
        if self.client.inspect_image(LLM_IMAGE_NAME).await.is_ok() {
            log::info!("LLM image already exists");
            return Ok(());
        }

        log::info!("Pulling LLM image...");

        // For now, we'll need to build from local Dockerfile
        // In production, this would pull from a registry
        Err(DockerError::ImageError(
            "LLM image not found. Please build it first with: docker compose -f docker/llm/docker-compose.yml build".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Docker to be running
    async fn test_docker_connection() {
        let manager = DockerManager::new().await;
        assert!(manager.is_ok(), "Should connect to Docker");
    }

    #[tokio::test]
    #[ignore] // Requires Docker to be running
    async fn test_check_status() {
        let manager = DockerManager::new().await.unwrap();
        let status = manager.check_status().await;
        assert!(status.available, "Docker should be available");
    }
}
