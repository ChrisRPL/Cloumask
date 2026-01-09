# Rust-Python HTTP Bridge

> **Parent:** 01-foundation
> **Depends on:** 06-fastapi-app-health, 09-tauri-ipc-commands
> **Blocks:** 11-frontend-ipc-utils, 12-dev-workflow-ollama

## Objective

Implement the HTTP client in Rust that communicates with the Python sidecar, enabling the Tauri app to proxy requests to FastAPI endpoints.

## Acceptance Criteria

- [ ] `reqwest` HTTP client integrated in sidecar module
- [ ] `call_sidecar<T>()` generic function for type-safe requests
- [ ] `check_health` command calls Python `/health` endpoint
- [ ] Connection errors handled gracefully with retries
- [ ] Timeout configuration for long-running operations
- [ ] JSON request/response serialization works correctly

## Implementation Steps

1. **Update Cargo.toml for reqwest**
   Ensure `src-tauri/Cargo.toml` has:
   ```toml
   [dependencies]
   reqwest = { version = "0.12", features = ["json"] }
   tokio = { version = "1", features = ["full"] }
   ```

2. **Add HTTP client to sidecar module**
   Update `src-tauri/src/sidecar.rs` to add HTTP functionality:
   ```rust
   //! Python sidecar process management and HTTP communication.

   use std::process::{Child, Command, Stdio};
   use std::sync::Mutex;
   use std::time::Duration;
   use std::thread;

   use reqwest::Client;
   use serde::{de::DeserializeOwned, Deserialize, Serialize};
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

       #[error("HTTP request failed: {0}")]
       HttpError(#[from] reqwest::Error),

       #[error("Sidecar health check failed: {0}")]
       HealthCheckFailed(String),

       #[error("Request timeout after {0}ms")]
       Timeout(u64),
   }

   /// Health response from the sidecar.
   #[derive(Debug, Serialize, Deserialize, Clone)]
   pub struct HealthResponse {
       pub status: String,
       pub version: String,
       pub timestamp: String,
       pub components: std::collections::HashMap<String, String>,
   }

   /// Configuration for the sidecar process.
   #[derive(Debug, Clone)]
   pub struct SidecarConfig {
       pub host: String,
       pub port: u16,
       pub python_path: String,
       pub backend_path: String,
       pub debug: bool,
       /// Timeout for HTTP requests in milliseconds
       pub request_timeout_ms: u64,
       /// Number of retry attempts for failed requests
       pub max_retries: u32,
   }

   impl Default for SidecarConfig {
       fn default() -> Self {
           Self {
               host: "127.0.0.1".to_string(),
               port: 8765,
               python_path: "python3".to_string(),
               backend_path: "backend".to_string(),
               debug: cfg!(debug_assertions),
               request_timeout_ms: 30000,  // 30 seconds
               max_retries: 3,
           }
       }
   }

   /// Manages the Python sidecar process lifecycle and HTTP communication.
   pub struct SidecarManager {
       process: Mutex<Option<Child>>,
       config: SidecarConfig,
       client: Client,
   }

   impl SidecarManager {
       /// Create a new sidecar manager with the given configuration.
       pub fn new(config: SidecarConfig) -> Self {
           let client = Client::builder()
               .timeout(Duration::from_millis(config.request_timeout_ms))
               .build()
               .expect("Failed to create HTTP client");

           Self {
               process: Mutex::new(None),
               config,
               client,
           }
       }

       /// Create a new sidecar manager with default configuration.
       pub fn with_defaults() -> Self {
           Self::new(SidecarConfig::default())
       }

       // ... (keep existing spawn, kill, is_running methods) ...

       /// Make a GET request to the sidecar.
       pub async fn get<T: DeserializeOwned>(&self, endpoint: &str) -> Result<T, SidecarError> {
           let url = format!("{}{}", self.base_url(), endpoint);
           self.request_with_retry(|| async {
               self.client.get(&url).send().await
           })
           .await?
           .json::<T>()
           .await
           .map_err(SidecarError::from)
       }

       /// Make a POST request to the sidecar with JSON body.
       pub async fn post<T, B>(&self, endpoint: &str, body: &B) -> Result<T, SidecarError>
       where
           T: DeserializeOwned,
           B: Serialize,
       {
           let url = format!("{}{}", self.base_url(), endpoint);
           self.request_with_retry(|| async {
               self.client.post(&url).json(body).send().await
           })
           .await?
           .json::<T>()
           .await
           .map_err(SidecarError::from)
       }

       /// Execute a request with retry logic.
       async fn request_with_retry<F, Fut>(
           &self,
           request_fn: F,
       ) -> Result<reqwest::Response, SidecarError>
       where
           F: Fn() -> Fut,
           Fut: std::future::Future<Output = Result<reqwest::Response, reqwest::Error>>,
       {
           let mut last_error = None;

           for attempt in 0..self.config.max_retries {
               match request_fn().await {
                   Ok(response) if response.status().is_success() => {
                       return Ok(response);
                   }
                   Ok(response) => {
                       last_error = Some(SidecarError::HealthCheckFailed(format!(
                           "HTTP {} - {}",
                           response.status(),
                           response.status().canonical_reason().unwrap_or("Unknown")
                       )));
                   }
                   Err(e) => {
                       last_error = Some(SidecarError::HttpError(e));
                   }
               }

               // Exponential backoff: 100ms, 200ms, 400ms, ...
               if attempt < self.config.max_retries - 1 {
                   let delay = Duration::from_millis(100 * (1 << attempt));
                   tokio::time::sleep(delay).await;
               }
           }

           Err(last_error.unwrap_or(SidecarError::HttpError(
               reqwest::Error::from(std::io::Error::new(
                   std::io::ErrorKind::Other,
                   "Request failed after retries",
               )),
           )))
       }

       /// Check the health of the sidecar.
       pub async fn health_check(&self) -> Result<HealthResponse, SidecarError> {
           self.get::<HealthResponse>("/health").await
       }

       /// Wait for the sidecar to become healthy.
       pub async fn wait_for_ready(&self, timeout: Duration) -> Result<(), SidecarError> {
           let start = std::time::Instant::now();
           let poll_interval = Duration::from_millis(200);

           while start.elapsed() < timeout {
               match self.health_check().await {
                   Ok(health) if health.status == "healthy" => {
                       return Ok(());
                   }
                   _ => {
                       tokio::time::sleep(poll_interval).await;
                   }
               }
           }

           Err(SidecarError::Timeout(timeout.as_millis() as u64))
       }

       /// Get the sidecar's base URL.
       pub fn base_url(&self) -> String {
           format!("http://{}:{}", self.config.host, self.config.port)
       }

       /// Get the sidecar's port.
       pub fn port(&self) -> u16 {
           self.config.port
       }
   }
   ```

3. **Add health check command**
   Update `src-tauri/src/commands/sidecar.rs`:
   ```rust
   //! Sidecar-related IPC commands.

   use serde::{Deserialize, Serialize};
   use tauri::State;

   use crate::sidecar::HealthResponse;
   use crate::state::AppState;

   // ... (keep existing SidecarStatus struct and commands) ...

   /// Check the health of the Python sidecar by calling its /health endpoint.
   #[tauri::command]
   pub async fn check_health(state: State<'_, AppState>) -> Result<HealthResponse, String> {
       state
           .sidecar
           .health_check()
           .await
           .map_err(|e| format!("Health check failed: {}", e))
   }

   /// Call a generic sidecar endpoint (for debugging).
   #[tauri::command]
   pub async fn call_sidecar_get(
       state: State<'_, AppState>,
       endpoint: String,
   ) -> Result<serde_json::Value, String> {
       state
           .sidecar
           .get::<serde_json::Value>(&endpoint)
           .await
           .map_err(|e| format!("Sidecar request failed: {}", e))
   }
   ```

4. **Register new commands in lib.rs**
   Update `src-tauri/src/lib.rs`:
   ```rust
   use commands::{
       call_sidecar_get, check_health, echo, get_app_info, get_sidecar_status,
       ping, restart_sidecar, start_sidecar, stop_sidecar,
   };

   // In invoke_handler:
   .invoke_handler(tauri::generate_handler![
       // System commands
       ping,
       echo,
       get_app_info,
       // Sidecar commands
       get_sidecar_status,
       start_sidecar,
       stop_sidecar,
       restart_sidecar,
       check_health,
       call_sidecar_get,
   ])
   ```

5. **Update sidecar startup to wait for ready**
   Update the setup hook in `lib.rs`:
   ```rust
   .setup(|app| {
       let config = SidecarConfig {
           python_path: "python3".to_string(),
           backend_path: resolve_backend_path(app),
           ..Default::default()
       };

       let sidecar = SidecarManager::new(config);
       if let Err(e) = sidecar.spawn() {
           eprintln!("Warning: Failed to spawn sidecar: {}", e);
       }

       // Store state first so we can use it
       let state = AppState::new(sidecar);
       app.manage(state);

       // Wait for sidecar in background (don't block app startup)
       let sidecar_clone = app.state::<AppState>().sidecar.clone();
       tauri::async_runtime::spawn(async move {
           match sidecar_clone
               .wait_for_ready(std::time::Duration::from_secs(10))
               .await
           {
               Ok(()) => println!("Sidecar is ready"),
               Err(e) => eprintln!("Warning: Sidecar not ready: {}", e),
           }
       });

       println!("Cloumask initialized successfully");
       Ok(())
   })
   ```

6. **Update TypeScript types**
   Update `src/lib/types/commands.ts`:
   ```typescript
   export interface HealthResponse {
     status: 'healthy' | 'degraded' | 'unhealthy';
     version: string;
     timestamp: string;
     components: Record<string, string>;
   }

   export interface TauriCommands {
     // ... existing commands ...

     // Health commands
     check_health(): Promise<HealthResponse>;
     call_sidecar_get(endpoint: string): Promise<unknown>;
   }
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src-tauri/Cargo.toml` | Verify | reqwest and tokio dependencies |
| `src-tauri/src/sidecar.rs` | Modify | Add HTTP client methods |
| `src-tauri/src/commands/sidecar.rs` | Modify | Add check_health command |
| `src-tauri/src/lib.rs` | Modify | Register new commands |
| `src/lib/types/commands.ts` | Modify | Add HealthResponse type |

## Verification

```bash
# Start the full application
cargo tauri dev

# Make sure Python sidecar is running first:
# In terminal: curl http://localhost:8765/health

# Test via Tauri IPC in browser devtools:
await window.__TAURI__.core.invoke('check_health')
// Should return: { status: "healthy", version: "0.1.0", ... }

# Test generic endpoint call:
await window.__TAURI__.core.invoke('call_sidecar_get', { endpoint: '/ready' })
// Should return: { ready: true, checks: {...} }

# Test error handling (stop sidecar first):
await window.__TAURI__.core.invoke('stop_sidecar')
await window.__TAURI__.core.invoke('check_health')
// Should return error: "Health check failed: ..."
```

## Notes

- The HTTP client uses `reqwest` with JSON feature for easy serialization
- Retry logic with exponential backoff handles transient failures
- Timeout is configurable (default 30s for long CV operations)
- The `wait_for_ready` method polls until sidecar responds
- All HTTP errors are converted to user-friendly error strings
- The generic `call_sidecar_get` command is useful for debugging
- Consider adding SSE support later for streaming responses
