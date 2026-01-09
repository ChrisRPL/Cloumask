# Tauri IPC Command Handlers

> **Parent:** 01-foundation
> **Depends on:** 08-rust-sidecar-module
> **Blocks:** 10-rust-python-http, 11-frontend-ipc-utils

## Objective

Create Tauri IPC commands that expose sidecar functionality to the frontend via the `invoke()` API.

## Acceptance Criteria

- [ ] `commands/` module created with command handlers
- [ ] `get_sidecar_status` command returns sidecar state
- [ ] `check_health` command verifies sidecar health
- [ ] Commands registered in `main.rs` invoke handler
- [ ] All commands return `Result<T, String>` for error handling
- [ ] TypeScript types generated or documented

## Implementation Steps

1. **Create the commands module directory**
   ```bash
   mkdir -p src-tauri/src/commands
   ```

2. **Create the main commands module**
   Create `src-tauri/src/commands/mod.rs`:
   ```rust
   //! Tauri IPC command handlers.
   //!
   //! All commands follow the pattern:
   //! - Accept `State<AppState>` for shared state access
   //! - Return `Result<T, String>` for proper error handling
   //! - Use descriptive names matching frontend expectations

   mod sidecar;
   mod system;

   pub use sidecar::*;
   pub use system::*;
   ```

3. **Create sidecar commands**
   Create `src-tauri/src/commands/sidecar.rs`:
   ```rust
   //! Sidecar-related IPC commands.

   use serde::{Deserialize, Serialize};
   use tauri::State;

   use crate::state::AppState;

   /// Response for sidecar status query.
   #[derive(Debug, Serialize, Deserialize)]
   pub struct SidecarStatus {
       /// Whether the sidecar process is running
       pub running: bool,
       /// The base URL of the sidecar
       pub url: String,
       /// The port the sidecar is listening on
       pub port: u16,
   }

   /// Get the current status of the Python sidecar.
   #[tauri::command]
   pub async fn get_sidecar_status(state: State<'_, AppState>) -> Result<SidecarStatus, String> {
       Ok(SidecarStatus {
           running: state.sidecar.is_running(),
           url: state.sidecar.base_url(),
           port: state.sidecar.port(),
       })
   }

   /// Start the Python sidecar if not already running.
   #[tauri::command]
   pub async fn start_sidecar(state: State<'_, AppState>) -> Result<String, String> {
       if state.sidecar.is_running() {
           return Ok("Sidecar already running".to_string());
       }

       state
           .sidecar
           .spawn()
           .map(|_| "Sidecar started successfully".to_string())
           .map_err(|e| format!("Failed to start sidecar: {}", e))
   }

   /// Stop the Python sidecar.
   #[tauri::command]
   pub async fn stop_sidecar(state: State<'_, AppState>) -> Result<String, String> {
       state
           .sidecar
           .kill()
           .map(|_| "Sidecar stopped successfully".to_string())
           .map_err(|e| format!("Failed to stop sidecar: {}", e))
   }

   /// Restart the Python sidecar.
   #[tauri::command]
   pub async fn restart_sidecar(state: State<'_, AppState>) -> Result<String, String> {
       // Kill existing process (ignore errors if not running)
       let _ = state.sidecar.kill();

       // Start fresh
       state
           .sidecar
           .spawn()
           .map(|_| "Sidecar restarted successfully".to_string())
           .map_err(|e| format!("Failed to restart sidecar: {}", e))
   }
   ```

4. **Create system commands**
   Create `src-tauri/src/commands/system.rs`:
   ```rust
   //! System-level IPC commands.

   use serde::{Deserialize, Serialize};

   /// Response for app info query.
   #[derive(Debug, Serialize, Deserialize)]
   pub struct AppInfo {
       /// Application name
       pub name: String,
       /// Application version
       pub version: String,
       /// Tauri version
       pub tauri_version: String,
       /// Debug mode enabled
       pub debug: bool,
   }

   /// Get application information.
   #[tauri::command]
   pub async fn get_app_info() -> Result<AppInfo, String> {
       Ok(AppInfo {
           name: "Cloumask".to_string(),
           version: env!("CARGO_PKG_VERSION").to_string(),
           tauri_version: tauri::VERSION.to_string(),
           debug: cfg!(debug_assertions),
       })
   }

   /// Simple ping command for testing IPC connectivity.
   #[tauri::command]
   pub async fn ping() -> Result<String, String> {
       Ok("pong".to_string())
   }

   /// Echo back a message (for testing).
   #[tauri::command]
   pub async fn echo(message: String) -> Result<String, String> {
       Ok(format!("Echo: {}", message))
   }
   ```

5. **Update lib.rs to export commands**
   Update `src-tauri/src/lib.rs`:
   ```rust
   //! Cloumask Tauri application library.

   pub mod commands;
   pub mod sidecar;
   pub mod state;

   use commands::{
       echo, get_app_info, get_sidecar_status, ping,
       restart_sidecar, start_sidecar, stop_sidecar,
   };
   use sidecar::{SidecarConfig, SidecarManager};
   use state::AppState;
   use std::thread;
   use std::time::Duration;

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
               }

               // Wait for sidecar to be ready
               thread::sleep(Duration::from_secs(2));

               // Store state
               app.manage(AppState::new(sidecar));

               println!("Cloumask initialized successfully");
               Ok(())
           })
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
           ])
           .on_window_event(|window, event| {
               if let tauri::WindowEvent::CloseRequested { .. } = event {
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
       if cfg!(debug_assertions) {
           std::env::current_dir()
               .map(|p| p.join("backend").to_string_lossy().to_string())
               .unwrap_or_else(|_| "backend".to_string())
       } else {
           app.path()
               .resource_dir()
               .map(|p| p.join("backend").to_string_lossy().to_string())
               .unwrap_or_else(|_| "backend".to_string())
       }
   }
   ```

6. **Create TypeScript type definitions**
   Create `src/lib/types/commands.ts` (for documentation):
   ```typescript
   // Tauri IPC command types
   // These match the Rust command response types

   export interface SidecarStatus {
     running: boolean;
     url: string;
     port: number;
   }

   export interface AppInfo {
     name: string;
     version: string;
     tauri_version: string;
     debug: boolean;
   }

   // Command function signatures
   export interface TauriCommands {
     // System commands
     ping(): Promise<string>;
     echo(message: string): Promise<string>;
     get_app_info(): Promise<AppInfo>;

     // Sidecar commands
     get_sidecar_status(): Promise<SidecarStatus>;
     start_sidecar(): Promise<string>;
     stop_sidecar(): Promise<string>;
     restart_sidecar(): Promise<string>;
   }
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src-tauri/src/commands/mod.rs` | Create | Command module exports |
| `src-tauri/src/commands/sidecar.rs` | Create | Sidecar IPC commands |
| `src-tauri/src/commands/system.rs` | Create | System IPC commands |
| `src-tauri/src/lib.rs` | Modify | Register commands |
| `src/lib/types/commands.ts` | Create | TypeScript type definitions |

## Verification

```bash
# Build and run
cargo tauri dev

# In browser devtools console, test IPC:
await window.__TAURI__.core.invoke('ping')
// Should return: "pong"

await window.__TAURI__.core.invoke('echo', { message: 'Hello' })
// Should return: "Echo: Hello"

await window.__TAURI__.core.invoke('get_app_info')
// Should return: { name: "Cloumask", version: "0.1.0", ... }

await window.__TAURI__.core.invoke('get_sidecar_status')
// Should return: { running: true, url: "http://127.0.0.1:8765", port: 8765 }
```

## Notes

- All async commands use `async fn` and return `Result<T, String>`
- The `State<'_, AppState>` parameter provides access to shared state
- Commands are registered in `invoke_handler` with `generate_handler!` macro
- Error messages should be user-friendly strings
- The TypeScript types serve as documentation for frontend developers
- Consider using `tauri-specta` for automatic TypeScript generation in the future
