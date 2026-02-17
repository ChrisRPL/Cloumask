# Cloumask Architecture + Tech-Fit Audit

Date: 2026-02-17  
Owner: BUILDER-A (docs-only pass)

## Scope

- As-built architecture map (from current code/manifests)
- Use-case -> tech-fit matrix (keep/adjust/remove recommendations)
- Desktop app keep/drop decision record
- Cleanup candidate inventory (unused imports/libs/folders/packages) with confidence + risk
- Prioritized cleanup plan for follow-on builders

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

## 2) Use-Case -> Tech-Fit Matrix

| Use Case | Active Path | Current Tech | Fit | Recommendation | Rationale |
|---|---|---|---|---|---|
| Conversational planning + step execution | Frontend chat -> FastAPI SSE -> LangGraph -> LangChain/Ollama | `backend/api/streaming`, `backend/agent/graph.py`, `backend/agent/llm/provider.py` | High | Keep | Node graph + HITL checkpoints match product workflow and are already test-covered. |
| Dataset review + annotation | Svelte review UI + Annotorious + backend review routes | `src/lib/components/Review/*`, `@annotorious/annotorious`, FastAPI routes | High | Keep | Clean split between UI and API; local-first review loop is functional. |
| Local desktop runtime | Tauri shell + Rust sidecar manager + IPC bridge | `src-tauri/src/lib.rs`, `src-tauri/src/sidecar.rs`, `src/lib/utils/tauri.ts` | High | Keep | Desktop mode gives bundled local runtime and native integration without server ops. |
| Point cloud inspect/transform | Three.js viewer + Rust native I/O/convert + Python CV routes | `src/lib/utils/three/*`, `src-tauri/src/pointcloud/*`, `backend/api/routes/pointcloud.py` | Medium-High | Keep (tighten contracts) | Hybrid Rust+Python stack is justified by performance and model tooling needs. |
| LLM model readiness + pull UX | Backend LLM routes + Tauri/web client actions | `backend/api/routes/llm.py`, `src/lib/utils/tauri.ts`, `src-tauri/src/commands/llm.rs` | Medium | Keep (monitor) | Works for local-first usage; sensitive to Ollama health/model availability. |
| Multi-user/shared infra mode | Local Ollama default + local sidecar assumptions | `backend/api/config.py`, local host defaults | Medium-Low | Adjust | Current defaults are single-host oriented; shared mode needs explicit provider/runtime policy. |
| Dependency/install reliability | Mixed `pyproject.toml` + `requirements*.txt` + root scripts | `backend/pyproject.toml`, `backend/requirements*.txt`, `package.json` | Medium-Low | Adjust | Multiple install paths can drift and produce inconsistent runtime capability. |

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

## 4) Cleanup Candidate Inventory (Docs-Only Baseline)

| Candidate | Type | Evidence | Confidence | Risk if Removed/Changed |
|---|---|---|---|---|
| `@internationalized/date` in frontend deps | Unused package | Present in `package.json`; no source imports found in `src/` or `tests/` | High | Low |
| `src-tauri/src/docker.rs` + `bollard` crate | Unused module/dependency path | `mod docker;` exists, but no command wiring or call sites outside module/tests | High | Medium |
| Python dependency source-of-truth drift (`pyproject.toml` vs `requirements*.txt`) | Duplicate manifests | Same capabilities declared in multiple files with non-identical package sets/versions | High | Medium |
| Backend install path mismatch (`pip install -e .` without extras) | Install/runtime gap | Root `backend:install` script installs core package only; agent/CV routes require optional deps | Medium | Medium-High |
| Dual model scaffold roots (`models/` and `backend/models/`) | Folder duplication | Both roots contain model READMEs/scaffolds; runtime default can vary by working directory | Medium | Medium |

## 5) Prioritized Cleanup Plan (for B/C)

1. `P1` Validate `src-tauri/src/docker.rs` usage; either wire into command surface or remove module + `bollard` safely (BUILDER-B).
2. `P1` Pick one Python dependency authority (`pyproject` or lockfile flow), then make `backend:install` deterministic for agent+CV runtime (BUILDER-B).
3. `P2` Remove confirmed-unused frontend dependency `@internationalized/date` with `npm run check` + `npm test -- --run` validation (BUILDER-C).
4. `P2` Normalize model-root policy (`backend/models` vs repo `models`) and document exact runtime expectation (BUILDER-B/C).
5. `P3` After above, run a second unused-dependency sweep and only remove items with passing full gate (`just ci`) (BUILDER-B/C).

## 6) Evidence Pointers

- Frontend/Tauri bridge: `src/lib/utils/tauri.ts`
- Desktop app bootstrap + command registry: `src-tauri/src/lib.rs`
- Sidecar process manager: `src-tauri/src/sidecar.rs`
- Backend app/router registration: `backend/src/backend/api/main.py`
- Streaming SSE API: `backend/src/backend/api/streaming/endpoints.py`
- Agent graph: `backend/src/backend/agent/graph.py`
- Ollama provider/config: `backend/src/backend/agent/llm/provider.py`, `backend/src/backend/agent/llm/config.py`
- Runtime/dependency manifests: `package.json`, `backend/pyproject.toml`, `backend/requirements*.txt`, `src-tauri/Cargo.toml`
- Desktop Docker candidate path: `src-tauri/src/docker.rs`
