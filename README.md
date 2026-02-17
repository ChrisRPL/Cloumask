# Cloumask

Cloumask is a local-first computer-vision workbench for dataset prep and review.

It combines:
- A Svelte 5 frontend (`src/`)
- A Rust/Tauri shell (`src-tauri/`) for desktop packaging and native bridge commands
- A FastAPI + LangGraph backend sidecar (`backend/`) using local Ollama models

## Architecture

```text
Web mode
  Browser (SvelteKit) ──HTTP/SSE──> FastAPI sidecar ──> LangGraph agent ──> Ollama

Desktop mode
  Tauri shell + Svelte UI ──IPC/HTTP──> Rust sidecar manager ──> FastAPI sidecar
                                                  └──────────> native point-cloud commands
```

Reference: `docs/ARCHITECTURE_TECH_FIT_AUDIT.md`

## Repository Layout

```text
src/                 Svelte frontend
src-tauri/           Rust/Tauri desktop shell + sidecar management
backend/             FastAPI API, agent graph, CV/data tools
docs/                Architecture, plans, testing docs
tests/               Playwright end-to-end tests
```

## Install

### Prerequisites

- Node.js 20+
- Python 3.11+
- Rust (for desktop builds/dev)
- Ollama (for local model-backed agent flows)

### Setup

```bash
npm install
npm run backend:install
```

## Run

### Web app (recommended for dev loop)

```bash
# Terminal 1
npm run backend:dev

# Terminal 2
npm run dev
```

Open `http://localhost:5173`.

### Desktop app

```bash
npm run tauri:dev
```

### Build desktop artifacts

```bash
npm run tauri:build
```

## Verify

Project gate:

```bash
just ci
```

Equivalent manual checks:

```bash
npm run check
npm test -- --run
cd backend && PYTHONPATH=src pytest -q
cd src-tauri && cargo test
```

## Troubleshooting

### `dyld: Library not loaded ... libsimdjson.26.dylib`

Your Node runtime is linked against a missing Homebrew library.

1. Reinstall Node (or your version manager runtime).
2. Reinstall/update `simdjson` via Homebrew.
3. Re-run `npm install`.

### Backend not reachable on `127.0.0.1:8765`

- Confirm backend is running: `npm run backend:dev`
- Check health endpoint: `curl -fsS http://127.0.0.1:8765/health`

### Ollama/model errors

- Ensure Ollama daemon is running.
- Confirm model availability in Ollama.
- Optional model override:

```bash
CLOUMASK_OLLAMA_MODEL=qwen3:14b npm run backend:dev
```

### Port conflicts

Default ports:
- Frontend: `5173`
- Backend: `8765`
- Ollama: `11434`

Free the conflicting process or stop existing dev sessions.

## Docs

- `docs/ARCHITECTURE_TECH_FIT_AUDIT.md`
- `docs/plan/README.md`
- `docs/testing/PLAYWRIGHT_FULL_QA_SCENARIOS.md`
- `backend/README.md`
- `CONTRIBUTING.md`

## License

MIT
