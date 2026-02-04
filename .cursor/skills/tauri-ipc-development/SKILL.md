---
name: tauri-ipc-development
description: Guide for creating Tauri IPC commands in Rust following Cloumask patterns. Use when adding new frontend-backend communication, creating Tauri commands, or debugging IPC issues.
---

# Tauri IPC Development

## Quick Start

When creating a new Tauri command:

1. Create command function with `#[tauri::command]` attribute
2. Return `Result<T, String>` for error handling
3. Use `State<'_, AppState>` for accessing app state
4. Use `async fn` for I/O operations
5. Export in `commands/mod.rs`

## Command Structure

All Tauri commands follow this pattern:

```rust
use tauri::State;
use crate::state::AppState;

/// Brief description of what this command does.
///
/// More detailed explanation if needed.
#[tauri::command]
pub fn my_command(state: State<'_, AppState>, param: String) -> Result<String, String> {
    // Implementation
    Ok("result".to_string())
}
```

## Async Commands

For I/O operations, use async commands:

```rust
use tauri::State;
use crate::state::AppState;

#[tauri::command]
pub async fn read_file_async(
    state: State<'_, AppState>,
    path: String,
) -> Result<String, String> {
    tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))
}
```

## Blocking Operations

For CPU-intensive or blocking operations, use `spawn_blocking`:

```rust
use tauri::State;

#[tauri::command]
pub async fn process_data(
    state: State<'_, AppState>,
    data: Vec<u8>,
) -> Result<ProcessedData, String> {
    tokio::task::spawn_blocking(move || {
        // CPU-intensive work here
        process_cpu_intensive(data)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| format!("Processing failed: {}", e))
}
```

## Error Handling

Always return `Result<T, String>` with descriptive error messages:

```rust
#[tauri::command]
pub fn my_command(path: String) -> Result<Data, String> {
    // Validate input
    if path.is_empty() {
        return Err("Path cannot be empty".to_string());
    }
    
    // Handle errors with context
    let data = std::fs::read(&path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;
    
    Ok(data)
}
```

## Accessing App State

Use `State<'_, AppState>` to access shared state:

```rust
use tauri::State;
use crate::state::AppState;
use crate::sidecar::Sidecar;

#[tauri::command]
pub fn get_sidecar_status(state: State<'_, AppState>) -> Result<SidecarStatus, String> {
    Ok(SidecarStatus {
        running: state.sidecar.is_running(),
        url: state.sidecar.base_url(),
    })
}
```

## Emitting Events

Use `AppHandle` to emit events to frontend:

```rust
use tauri::{AppHandle, Emitter};

#[tauri::command]
pub async fn stream_data(
    app: AppHandle,
    path: String,
) -> Result<Metadata, String> {
    // Emit events asynchronously
    tokio::spawn(async move {
        for chunk in stream_chunks(&path).await {
            app.emit("data:chunk", chunk).unwrap();
        }
        app.emit("data:complete", ()).unwrap();
    });
    
    Ok(metadata)
}
```

## Command Organization

Organize commands by domain in separate modules:

```
src-tauri/src/commands/
├── mod.rs          # Re-exports all commands
├── system.rs       # System info, ping, echo
├── sidecar.rs      # Sidecar management
├── pointcloud.rs   # Point cloud operations
└── llm.rs          # LLM service queries
```

Export in `mod.rs`:

```rust
//! Tauri IPC command handlers.

mod system;
mod sidecar;
mod pointcloud;
mod llm;

pub use system::*;
pub use sidecar::*;
pub use pointcloud::*;
pub use llm::*;
```

## Registering Commands

Register commands in `main.rs`:

```rust
use tauri::Builder;

fn main() {
    Builder::default()
        .invoke_handler(tauri::generate_handler![
            // System commands
            get_app_info,
            ping,
            echo,
            // Sidecar commands
            sidecar_status,
            start_sidecar,
            stop_sidecar,
            // Point cloud commands
            read_pointcloud_metadata,
            read_pointcloud,
            stream_pointcloud,
            // LLM commands
            get_llm_status,
            list_llm_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

## Type Serialization

Use `serde` for custom types:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyData {
    pub field1: String,
    pub field2: i32,
}

#[tauri::command]
pub fn get_data() -> Result<MyData, String> {
    Ok(MyData {
        field1: "value".to_string(),
        field2: 42,
    })
}
```

## Frontend Invocation

Commands are invoked from TypeScript/Svelte:

```typescript
import { invoke } from '@tauri-apps/api/core';

// Simple command
const result = await invoke<string>('my_command', { param: 'value' });

// With error handling
try {
    const data = await invoke<MyData>('get_data');
} catch (error) {
    console.error('Command failed:', error);
}
```

## Common Patterns

### File Operations

```rust
#[tauri::command]
pub async fn read_file(path: String) -> Result<Vec<u8>, String> {
    tokio::fs::read(&path)
        .await
        .map_err(|e| format!("Failed to read {}: {}", path, e))
}
```

### Sidecar HTTP Requests

```rust
use crate::state::AppState;

#[tauri::command]
pub async fn call_sidecar(
    state: State<'_, AppState>,
    endpoint: String,
) -> Result<serde_json::Value, String> {
    state
        .sidecar
        .get_async::<serde_json::Value>(&endpoint)
        .await
        .map_err(|e| format!("Sidecar request failed: {}", e))
}
```

### Progress Reporting

```rust
#[tauri::command]
pub async fn process_with_progress(
    app: AppHandle,
    input: String,
) -> Result<(), String> {
    let total = 100;
    for i in 0..total {
        // Emit progress
        app.emit("progress", ProgressUpdate {
            current: i,
            total,
            message: format!("Processing {}%", i),
        }).unwrap();
        
        // Do work
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    
    Ok(())
}
```

## Testing

Test commands with Tauri's test utilities:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_my_command() {
        let result = my_command("test".to_string());
        assert!(result.is_ok());
    }
}
```

## Best Practices

1. **Always return `Result<T, String>`** - Never panic, always return errors
2. **Use async for I/O** - Don't block the main thread
3. **Use `spawn_blocking` for CPU work** - Keep async runtime responsive
4. **Provide context in errors** - Include file paths, operation names
5. **Emit events for long operations** - Keep frontend informed
6. **Validate inputs early** - Fail fast with clear messages
7. **Use `State` for shared resources** - Don't create new connections per command

## Additional Resources

- See `src-tauri/src/commands/system.rs` for simple command examples
- See `src-tauri/src/commands/pointcloud.rs` for async streaming examples
- See `src-tauri/src/commands/sidecar.rs` for state access examples
