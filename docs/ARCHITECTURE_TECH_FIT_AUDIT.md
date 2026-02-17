# Cloumask Architecture + Tech-Fit Audit

Date: 2026-02-17  
Owner: BUILDER-A (docs-only pass)

## Scope

- As-built architecture map (from current code/manifests)
- Tech-fit audit (keep/adjust/drop recommendations)
- Desktop app keep/drop decision record

## 1) As-Built Architecture Map

```text
Svelte 5 frontend (src/)
  ├─ Web mode: calls backend HTTP/SSE directly
  └─ Desktop mode: calls Tauri IPC (`src/lib/utils/tauri.ts`)
            ↓
Rust shell + command layer (src-tauri/src/lib.rs, src-tauri/src/commands/*)
  ├─ Manages Python sidecar lifecycle (`src-tauri/src/sidecar.rs`)
  ├─ Proxies health/LLM/HTTP calls to sidecar
  └─ Runs native point-cloud commands (I/O/convert/decimate)
            ↓
Python sidecar API (backend/src/backend/api/main.py)
  ├─ Streaming chat API (`backend/src/backend/api/streaming/endpoints.py`)
  ├─ CV/data routes (detect, anonymize, review, pointcloud, fusion, rosbag)
  └─ LLM readiness/model pull routes
            ↓
LangGraph agent runtime (backend/src/backend/agent/graph.py)
            ↓
LangChain-Ollama provider (backend/src/backend/agent/llm/provider.py)
            ↓
Local Ollama API (default: http://localhost:11434)
```

## 2) Tech-Fit Audit

| Area | Current Tech | Fit | Recommendation | Why |
|---|---|---|---|---|
| Desktop shell | Tauri 2 + Rust | High | Keep | Native shell is already integrated with sidecar lifecycle, app state checks, and point-cloud native commands. |
| Frontend | Svelte 5 + Vite | High | Keep | Frontend supports both Tauri and browser mode; existing tests cover key user flows. |
| Backend API | FastAPI + Pydantic + SSE | High | Keep | Clear route separation, typed contracts, and SSE thread model for live agent updates. |
| Agent orchestration | LangGraph + checkpoints | High | Keep | Graph/node architecture is explicit and testable; checkpoint/approval flow matches product needs. |
| LLM integration | LangChain-Ollama (`ChatOllama`) | Medium-High | Keep (monitor) | Good tool-calling abstraction + retries/fallbacks; tied to local Ollama model availability and host health. |
| LLM serving | Ollama local-only default | Medium | Keep for local-first | Strong privacy/offline fit; weaker for multi-user throughput and shared infra. |
| Point-cloud processing | Mixed Rust + Python | Medium-High | Keep | Rust path gives native performance for I/O/transforms; Python path covers model-heavy ML flows. |
| Dependency mgmt (Python) | `pyproject.toml` + `requirements*.txt` | Medium | Adjust | Dual source of truth can drift; keep but enforce sync policy in release checklist. |

## 3) Desktop Need Decision Record

### DR-2026-02-17: Desktop Runtime

- Status: Accepted
- Decision: **Keep desktop app**; do **not** make desktop exclusive.

### Context

- Repo already supports two valid dev/runtime flows:
  - Desktop: `npm run tauri:dev`
  - Web + backend: `npm run backend:dev` + `npm run dev`
- Desktop path currently delivers:
  - Sidecar lifecycle management and health visibility in-app
  - Native command bridge for point-cloud operations
  - Local-first onboarding model (privacy/offline posture)
- Web path currently delivers:
  - Faster contributor loop
  - Better CI/headless ergonomics
  - Non-Tauri fallback behavior already present in UI

### Rationale

- Product goal is local-first CV workflows, not browser-only SaaS.
- Existing architecture already amortized desktop complexity in Rust sidecar manager and command surface.
- Removing desktop now would discard working integration value and force new orchestration work into web-only runtime.

### Consequences

- Keep both modes as first-class:
  - Desktop = packaged local runtime for end users.
  - Web+backend = development/CI/runtime fallback.
- New features should avoid unnecessary desktop-only coupling unless native capability is required.
- Point-cloud/file-system-heavy flows can stay desktop-optimized, but must fail clearly in web mode.

## 4) Evidence Pointers

- Frontend/Tauri bridge: `src/lib/utils/tauri.ts`
- Desktop app bootstrap + command registry: `src-tauri/src/lib.rs`
- Sidecar process manager: `src-tauri/src/sidecar.rs`
- Backend app/router registration: `backend/src/backend/api/main.py`
- Streaming SSE API: `backend/src/backend/api/streaming/endpoints.py`
- Agent graph: `backend/src/backend/agent/graph.py`
- Ollama provider/config: `backend/src/backend/agent/llm/provider.py`, `backend/src/backend/agent/llm/config.py`
- Runtime/dependency manifests: `package.json`, `backend/pyproject.toml`, `src-tauri/Cargo.toml`
