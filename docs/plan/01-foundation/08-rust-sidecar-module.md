# Rust Sidecar Management Module

> **Parent:** 01-foundation
> **Depends on:** 01-tauri-project-init, 07-python-dependencies
> **Blocks:** 09-tauri-ipc-commands, 10-rust-python-http

## Objective

Implement the Rust module that manages the Python sidecar process lifecycle - spawning on app start and terminating on app close.

## Acceptance Criteria

- [ ] `sidecar.rs` module created with `SidecarManager` struct
- [ ] `spawn_sidecar()` starts Python uvicorn process
- [ ] `kill_sidecar()` gracefully terminates the process
- [ ] `AppState` struct holds sidecar manager in Tauri state
- [ ] Sidecar starts automatically in Tauri `setup()` hook
- [ ] Sidecar stops automatically in `on_exit()` hook
- [ ] Process errors are handled gracefully

## Implementation Steps

1. **Create the sidecar module**
   Create `src-tauri/src/sidecar.rs`:
   ```rust
   //! Python sidecar process management.
   //!
   //! Handles spawning, monitoring, and terminating the Python FastAPI backend.

   use std::process::{Child, Command, Stdio};
   use std::sync::Mutex;
   use std::time::Duration;
   use std::thread;

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

       #[error("Sidecar health check failed: {0}")]
       HealthCheckFailed(String),
   }

   /// Configuration for the sidecar process.
   #[derive(Debug, Clone)]
   pub struct SidecarConfig {
       /// Host to bind the sidecar server
       pub host: String,
       /// Port for the sidecar server
       pub port: u16,
       /// Path to Python executable
       pub python_path: String,
       /// Path to the backend directory
       pub backend_path: String,
       /// Whether to enable debug mode
       pub debug: bool,
   }

   impl Default for SidecarConfig {
       fn default() -> Self {
           Self {
               host: "127.0.0.1".to_string(),
               port: 8765,
               python_path: "python3".to_string(),
               backend_path: "backend".to_string(),
               debug: cfg!(debug_assertions),
           }
       }
   }

   /// Manages the Python sidecar process lifecycle.
   pub struct SidecarManager {
       /// The running child process, if any
       process: Mutex<Option<Child>>,
       /// Configuration for the sidecar
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
       pub fn with_defaults() -> Self {
           Self::new(SidecarConfig::default())
       }

       /// Spawn the Python sidecar process.
       ///
       /// Returns Ok(()) if the process was spawned successfully.
       /// If a process is already running, it will be killed first.
       pub fn spawn(&self) -> Result<(), SidecarError> {
           let mut process_guard = self.process.lock().map_err(|_| SidecarError::LockError)?;

           // Kill existing process if running
           if let Some(mut existing) = process_guard.take() {
               let _ = existing.kill();
               let _ = existing.wait();
           }

           // Build the uvicorn command
           let child = Command::new(&self.config.python_path)
               .args([
                   "-m",
                   "uvicorn",
                   "api.main:app",
                   "--host",
                   &self.config.host,
                   "--port",
                   &self.config.port.to_string(),
               ])
               .current_dir(&self.config.backend_path)
               .stdout(Stdio::piped())
               .stderr(Stdio::piped())
               .spawn()?;

           println!(
               "Sidecar spawned with PID: {}",
               child.id()
           );

           *process_guard = Some(child);
           Ok(())
       }

       /// Kill the sidecar process gracefully.
       ///
       /// Attempts a graceful shutdown first, then forces termination.
       pub fn kill(&self) -> Result<(), SidecarError> {
           let mut process_guard = self.process.lock().map_err(|_| SidecarError::LockError)?;

           if let Some(mut child) = process_guard.take() {
               println!("Killing sidecar process (PID: {})...", child.id());

               // Try graceful shutdown first
               #[cfg(unix)]
               {
                   use std::os::unix::process::CommandExt;
                   // Send SIGTERM for graceful shutdown
                   unsafe {
                       libc::kill(child.id() as i32, libc::SIGTERM);
                   }
               }

               #[cfg(windows)]
               {
                   // On Windows, just kill the process
                   let _ = child.kill();
               }

               // Wait a bit for graceful shutdown
               thread::sleep(Duration::from_millis(500));

               // Force kill if still running
               match child.try_wait() {
                   Ok(Some(_)) => println!("Sidecar exited gracefully"),
                   Ok(None) => {
                       println!("Sidecar still running, forcing termination...");
                       let _ = child.kill();
                       let _ = child.wait();
                   }
                   Err(e) => println!("Error checking sidecar status: {}", e),
               }

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
                   Ok(None) => true,  // Still running
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

       /// Get the sidecar's port.
       pub fn port(&self) -> u16 {
           self.config.port
       }
   }

   impl Drop for SidecarManager {
       fn drop(&mut self) {
           // Ensure sidecar is killed when manager is dropped
           let _ = self.kill();
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
       }

       #[test]
       fn test_base_url() {
           let manager = SidecarManager::with_defaults();
           assert_eq!(manager.base_url(), "http://127.0.0.1:8765");
       }
   }
   ```

2. **Create the app state module**
   Create `src-tauri/src/state.rs`:
   ```rust
   //! Application state management.

   use std::sync::Arc;

   use crate::sidecar::SidecarManager;

   /// Global application state shared across Tauri commands.
   pub struct AppState {
       /// Manager for the Python sidecar process
       pub sidecar: Arc<SidecarManager>,
   }

   impl AppState {
       /// Create a new app state with a sidecar manager.
       pub fn new(sidecar: SidecarManager) -> Self {
           Self {
               sidecar: Arc::new(sidecar),
           }
       }
   }
   ```

3. **Update lib.rs with modules and setup**
   Update `src-tauri/src/lib.rs`:
   ```rust
   //! Cloumask Tauri application library.

   pub mod sidecar;
   pub mod state;

   use sidecar::{SidecarConfig, SidecarManager};
   use state::AppState;
   use std::time::Duration;
   use std::thread;

   /// Initialize and run the Tauri application.
   #[cfg_attr(mobile, tauri::mobile_entry_point)]
   pub fn run() {
       tauri::Builder::default()
           .plugin(tauri_plugin_shell::init())
           .setup(|app| {
               // Configure sidecar
               let config = SidecarConfig {
                   python_path: "python3".to_string(),
                   backend_path: resolve_backend_path(app),
                   ..Default::default()
               };

               // Create and spawn sidecar
               let sidecar = SidecarManager::new(config);
               if let Err(e) = sidecar.spawn() {
                   eprintln!("Warning: Failed to spawn sidecar: {}", e);
                   // Don't fail app startup, sidecar can be started later
               }

               // Wait for sidecar to be ready
               thread::sleep(Duration::from_secs(2));

               // Store state for use in commands
               app.manage(AppState::new(sidecar));

               println!("Cloumask initialized successfully");
               Ok(())
           })
           .on_window_event(|window, event| {
               if let tauri::WindowEvent::CloseRequested { .. } = event {
                   // Kill sidecar when window is closed
                   if let Some(state) = window.try_state::<AppState>() {
                       if let Err(e) = state.sidecar.kill() {
                           eprintln!("Warning: Failed to kill sidecar: {}", e);
                       }
                   }
               }
           })
           .run(tauri::generate_context!())
           .expect("error while running Cloumask");
   }

   /// Resolve the path to the backend directory.
   fn resolve_backend_path(app: &tauri::App) -> String {
       // In development, use the relative path
       if cfg!(debug_assertions) {
           // Go up from src-tauri to project root, then to backend
           std::env::current_dir()
               .map(|p| p.join("backend").to_string_lossy().to_string())
               .unwrap_or_else(|_| "backend".to_string())
       } else {
           // In production, backend is bundled as a resource
           app.path()
               .resource_dir()
               .map(|p| p.join("backend").to_string_lossy().to_string())
               .unwrap_or_else(|_| "backend".to_string())
       }
   }
   ```

4. **Update Cargo.toml with dependencies**
   Ensure `src-tauri/Cargo.toml` has:
   ```toml
   [dependencies]
   tauri = { version = "2", features = ["devtools"] }
   tauri-plugin-shell = "2"
   serde = { version = "1", features = ["derive"] }
   serde_json = "1"
   tokio = { version = "1", features = ["full"] }
   reqwest = { version = "0.12", features = ["json"] }
   thiserror = "1"

   [target.'cfg(unix)'.dependencies]
   libc = "0.2"
   ```

5. **Add sidecar permission to tauri.conf.json**
   Update `src-tauri/tauri.conf.json` to include shell permissions:
   ```json
   {
     "plugins": {
       "shell": {
         "open": true,
         "scope": [
           {
             "name": "python",
             "cmd": "python3",
             "args": true
           }
         ]
       }
     }
   }
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src-tauri/src/sidecar.rs` | Create | Sidecar process management |
| `src-tauri/src/state.rs` | Create | Application state |
| `src-tauri/src/lib.rs` | Modify | Wire up modules and setup |
| `src-tauri/Cargo.toml` | Modify | Add libc dependency |
| `src-tauri/tauri.conf.json` | Modify | Shell plugin permissions |

## Verification

```bash
# Build and run
cd /Users/krzysztof/Cloumask
cargo tauri dev

# Check that sidecar starts:
# - Look for "Sidecar spawned with PID: ..." in console
# - Check process: ps aux | grep uvicorn

# Test health endpoint:
curl http://localhost:8765/health

# Close the app window and verify:
# - "Killing sidecar process..." message appears
# - No orphaned Python processes: ps aux | grep uvicorn
```

## Notes

- The sidecar manager uses `Mutex` for thread-safe process handling
- SIGTERM is used on Unix for graceful shutdown (allows uvicorn to cleanup)
- The sidecar path is resolved differently in dev vs production
- If sidecar fails to spawn, the app continues (allows debugging)
- The `Drop` impl ensures cleanup even on unexpected termination
- Consider adding health check polling to verify sidecar is ready
