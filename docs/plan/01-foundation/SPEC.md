# Foundation Module

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** None (first module)

## Overview

Set up the core application shell: Tauri 2.0 desktop framework with Svelte 5 frontend and Python FastAPI sidecar. Establish cross-layer communication (IPC) and development workflow.

## Goals

- [ ] Working Tauri 2.0 + Svelte 5 desktop application
- [ ] Python sidecar that starts/stops with the app
- [ ] Bidirectional communication between all layers
- [ ] Development environment with hot reload
- [ ] Ollama integration verified

## Technical Design

### Stack
- **Desktop Shell:** Tauri 2.0 (Rust)
- **Frontend:** Svelte 5, Vite, Tailwind CSS, shadcn/ui
- **Backend:** Python 3.11+, FastAPI, uvicorn
- **Bundling:** PyInstaller for sidecar executable

### Communication Flow
```
Frontend (Svelte)
    │
    │ Tauri IPC (invoke)
    ▼
Rust Core (Tauri)
    │
    │ HTTP/SSE (localhost:8765)
    ▼
Python Sidecar (FastAPI)
```

### Sidecar Lifecycle
1. Tauri `setup()` hook spawns Python process
2. Store `Child` handle in app state
3. Health check endpoint `/health`
4. Tauri `on_exit()` kills sidecar

## Implementation Tasks

- [ ] **Project Initialization**
  - [ ] Create Tauri 2.0 project with Svelte template
  - [ ] Configure `tauri.conf.json` for sidecar
  - [ ] Set up Vite with Tailwind CSS
  - [ ] Install shadcn/ui components

- [ ] **Python Sidecar Setup**
  - [ ] Create `backend/` directory structure
  - [ ] Initialize FastAPI application
  - [ ] Add health check endpoint
  - [ ] Configure uvicorn for development
  - [ ] Create `requirements.txt`

- [ ] **Rust Sidecar Management**
  - [ ] Implement `sidecar.rs` module
  - [ ] Spawn Python on app start
  - [ ] Kill Python on app exit
  - [ ] Handle process errors gracefully

- [ ] **IPC Layer**
  - [ ] Create Tauri commands for sidecar communication
  - [ ] Implement SSE streaming from Python
  - [ ] Create TypeScript types for IPC
  - [ ] Add error handling and retries

- [ ] **Development Workflow**
  - [ ] Configure `cargo tauri dev` for hot reload
  - [ ] Set up Python auto-reload (uvicorn --reload)
  - [ ] Add npm scripts for common tasks
  - [ ] Verify Ollama connectivity

## Acceptance Criteria

- [ ] `cargo tauri dev` starts the full application
- [ ] Frontend can call Rust commands via IPC
- [ ] Rust can communicate with Python sidecar
- [ ] Sidecar starts automatically and stops on app close
- [ ] Hot reload works for Svelte and Python changes
- [ ] `ollama list` shows available models from the app

## Files to Create/Modify

```
src-tauri/
├── Cargo.toml              # Add dependencies
├── tauri.conf.json         # Sidecar configuration
├── src/
│   ├── main.rs             # App entry point
│   ├── lib.rs              # Module exports
│   ├── sidecar.rs          # Sidecar management
│   └── commands/
│       └── mod.rs          # IPC command handlers

src/
├── app.html                # Svelte entry
├── routes/
│   └── +page.svelte        # Main page
├── lib/
│   ├── components/
│   │   └── ui/             # shadcn components
│   └── utils/
│       └── tauri.ts        # IPC utilities

backend/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── main.py             # FastAPI app
│   └── routes/
│       └── health.py       # Health endpoint
├── requirements.txt
└── pyproject.toml
```

## Sub-Specs (expand later)

- `tauri-setup.md` - Detailed Tauri configuration
- `sidecar-lifecycle.md` - Process management details
- `ipc-protocol.md` - Communication protocol spec
- `dev-environment.md` - Development setup guide

---

## Detailed Task Specs

This module is broken into 12 atomic implementation specs. Each spec is independently implementable with clear acceptance criteria.

### Frontend Setup (01-04)

| Spec | Title | Description |
|------|-------|-------------|
| [01-tauri-project-init.md](01-tauri-project-init.md) | Tauri 2.0 Project Initialization | Create Tauri project with Svelte template, configure Cargo.toml and tauri.conf.json |
| [02-svelte-vite-config.md](02-svelte-vite-config.md) | Svelte 5 and Vite Configuration | Configure SvelteKit, Vite, and static adapter for Tauri |
| [03-tailwind-css-setup.md](03-tailwind-css-setup.md) | Tailwind CSS Integration | Install and configure Tailwind with design system colors |
| [04-shadcn-ui-components.md](04-shadcn-ui-components.md) | shadcn/ui Component Library | Add shadcn-svelte with Button, Card, Input components |

### Python Backend (05-07)

| Spec | Title | Description |
|------|-------|-------------|
| [05-python-backend-structure.md](05-python-backend-structure.md) | Python Backend Scaffolding | Create directory structure and pyproject.toml |
| [06-fastapi-app-health.md](06-fastapi-app-health.md) | FastAPI Application with Health Endpoint | Create FastAPI app with CORS and /health endpoint |
| [07-python-dependencies.md](07-python-dependencies.md) | Python Dependencies Management | Create requirements.txt with pinned versions |

### Rust Integration (08-10)

| Spec | Title | Description |
|------|-------|-------------|
| [08-rust-sidecar-module.md](08-rust-sidecar-module.md) | Rust Sidecar Management Module | Implement SidecarManager for process lifecycle |
| [09-tauri-ipc-commands.md](09-tauri-ipc-commands.md) | Tauri IPC Command Handlers | Create #[tauri::command] handlers for frontend |
| [10-rust-python-http.md](10-rust-python-http.md) | Rust-Python HTTP Bridge | Implement reqwest client with retry logic |

### Final Integration (11-12)

| Spec | Title | Description |
|------|-------|-------------|
| [11-frontend-ipc-utils.md](11-frontend-ipc-utils.md) | Frontend IPC Utilities | TypeScript wrappers and system status UI |
| [12-dev-workflow-ollama.md](12-dev-workflow-ollama.md) | Development Workflow and Ollama Integration | Configure npm scripts, verify HMR, add Ollama endpoints |

### Dependency Graph

```
Frontend:    01 → 02 → 03 → 04 ────────────────────┐
                                                   │
Python:      05 → 06 → 07 ────────────────────────┐│
                                                  ││
Rust/IPC:    08 → 09 → 10 → 11 ───────────────────┼┤
                                                  ││
                                                  ▼▼
Integration:                                      12
```

### Mapping to Task Groups

| Original Task Group | Covered By Specs |
|--------------------|------------------|
| Project Initialization | 01, 02, 03, 04 |
| Python Sidecar Setup | 05, 06, 07 |
| Rust Sidecar Management | 08, 10 |
| IPC Layer | 09, 10, 11 |
| Development Workflow | 12 |
