# Tauri 2.0 Project Initialization

> **Parent:** 01-foundation
> **Depends on:** None (first task)
> **Blocks:** 02-svelte-vite-config, 08-rust-sidecar-module

## Objective

Create a new Tauri 2.0 desktop application with the Svelte template as the foundation for Cloumask.

## Acceptance Criteria

- [ ] Tauri 2.0 project created with Svelte frontend
- [ ] `src-tauri/Cargo.toml` contains required dependencies
- [ ] `src-tauri/tauri.conf.json` configured with app metadata
- [ ] `cargo tauri dev` starts and displays an empty window
- [ ] Basic `main.rs` and `lib.rs` files in place

## Implementation Steps

1. **Create Tauri project**
   ```bash
   cd /Users/krzysztof/Cloumask
   npm create tauri-app@latest . -- --template svelte-ts --manager npm
   ```
   - Select Tauri 2.0 when prompted
   - Choose TypeScript + Svelte template

2. **Update Cargo.toml dependencies**
   Add to `src-tauri/Cargo.toml`:
   ```toml
   [dependencies]
   tauri = { version = "2", features = ["devtools"] }
   tauri-plugin-shell = "2"
   serde = { version = "1", features = ["derive"] }
   serde_json = "1"
   tokio = { version = "1", features = ["full"] }
   reqwest = { version = "0.12", features = ["json"] }
   thiserror = "1"

   [build-dependencies]
   tauri-build = { version = "2", features = [] }
   ```

3. **Configure tauri.conf.json**
   Update `src-tauri/tauri.conf.json`:
   ```json
   {
     "$schema": "https://schema.tauri.app/config/2",
     "productName": "Cloumask",
     "version": "0.1.0",
     "identifier": "com.cloumask.app",
     "build": {
       "beforeDevCommand": "npm run dev",
       "devUrl": "http://localhost:5173",
       "beforeBuildCommand": "npm run build",
       "frontendDist": "../build"
     },
     "app": {
       "withGlobalTauri": true,
       "windows": [
         {
           "title": "Cloumask",
           "width": 1280,
           "height": 800,
           "minWidth": 800,
           "minHeight": 600,
           "resizable": true,
           "fullscreen": false
         }
       ],
       "security": {
         "csp": null
       }
     },
     "bundle": {
       "active": true,
       "targets": "all",
       "icon": [
         "icons/32x32.png",
         "icons/128x128.png",
         "icons/128x128@2x.png",
         "icons/icon.icns",
         "icons/icon.ico"
       ]
     }
   }
   ```

4. **Set up main.rs entry point**
   Create `src-tauri/src/main.rs`:
   ```rust
   // Prevents additional console window on Windows in release
   #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

   fn main() {
       cloumask_lib::run()
   }
   ```

5. **Set up lib.rs**
   Create `src-tauri/src/lib.rs`:
   ```rust
   #[cfg_attr(mobile, tauri::mobile_entry_point)]
   pub fn run() {
       tauri::Builder::default()
           .plugin(tauri_plugin_shell::init())
           .run(tauri::generate_context!())
           .expect("error while running Cloumask");
   }
   ```

6. **Update Cargo.toml package name**
   ```toml
   [package]
   name = "cloumask"
   version = "0.1.0"
   edition = "2021"

   [lib]
   name = "cloumask_lib"
   crate-type = ["staticlib", "cdylib", "rlib"]
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src-tauri/Cargo.toml` | Create | Rust dependencies and package config |
| `src-tauri/tauri.conf.json` | Create | Tauri app configuration |
| `src-tauri/src/main.rs` | Create | Application entry point |
| `src-tauri/src/lib.rs` | Create | Library exports |
| `src-tauri/build.rs` | Create | Build script (generated) |

## Verification

```bash
# Install dependencies
cd /Users/krzysztof/Cloumask
npm install

# Start development server
cargo tauri dev
```

Expected: Window titled "Cloumask" opens with default Svelte content.

## Notes

- Tauri 2.0 uses a different configuration format than 1.x
- The `withGlobalTauri` option exposes `window.__TAURI__` for IPC
- The shell plugin is required for spawning the Python sidecar later
- Keep CSP null for now; will configure properly in production
