# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cloumask is a local-first, agentic desktop application for computer vision data processing. Conversational AI interface for anonymization, auto-labeling, and dataset operations on 2D images, videos, and 3D point clouds.

## Architecture

**Three-layer stack:**
1. **Tauri 2.0 Shell** (Rust) - File I/O, point cloud processing (pasture), sidecar management
2. **Svelte 5 Frontend** - Chat, Plan Editor, Execution View, Review Queue, Point Cloud Viewer (Three.js)
3. **Python Sidecar** (FastAPI + PyInstaller) - LangGraph agent, CV models

**Communication:** Frontend ↔ Rust (Tauri IPC) ↔ Python (HTTP/SSE on port 8765)

## Development Commands

```bash
# Start development
cargo tauri dev                           # Full app with hot reload
npm run dev                               # Frontend only
cd backend && uvicorn api.main:app --reload --port 8765  # Python sidecar

# Testing
cd src-tauri && cargo test               # Rust tests
cd backend && pytest tests/ -v           # Python tests
npm run test                             # Frontend tests

# Verification (run before commits)
cd src-tauri && cargo clippy -- -D warnings
cd backend && ruff check . && mypy .
npm run check && npm run lint

# Build
cargo tauri build                        # Production build
cd backend && pyinstaller backend.spec   # Bundle sidecar
```

## Directory Structure

```
src-tauri/src/
  commands/       # Tauri IPC handlers
  pointcloud/     # Rust point cloud (pasture)
  sidecar.rs      # Python process management

src/lib/
  components/     # Svelte components (Chat/, Plan/, Execution/, Review/, PointCloud/)
  stores/         # Svelte stores for state
  utils/          # Helpers

backend/
  api/            # FastAPI routes, SSE streaming
  agent/          # LangGraph state machine, tools, prompts
  cv/             # Model wrappers (detection, segmentation, anonymization)
  data/           # Format loaders/exporters
```

## Agent State Machine

```
understand → plan → await_approval → execute_step → checkpoint → complete
                         ↑                              │
                         └──────────────────────────────┘
```

Checkpoints trigger at: 10%, 25%, 50% progress OR confidence drop >15% OR error rate >5%

## Key Patterns

**Tauri Commands (Rust):**
```rust
#[tauri::command]
async fn process_data(state: State<'_, AppState>, path: String) -> Result<Data, String> {
    // Always return Result<T, String> for IPC
}
```

**LangGraph Tools (Python):**
```python
@tool
def detect_objects(image_path: str, confidence: float = 0.5) -> DetectionResult:
    """Tool docstring becomes the description for the LLM."""
    # Return structured data, not strings
```

**Svelte Stores:**
```typescript
// Use runes in Svelte 5
let count = $state(0);
let doubled = $derived(count * 2);
```

## Common Mistakes to Avoid

1. **Don't load CV models eagerly** - Load on first use, unload when done
2. **Don't block Tauri main thread** - Use async commands for I/O
3. **Don't forget SSE cleanup** - Close EventSource on component unmount
4. **Don't skip checkpoint approval** - Agent must pause for human review
5. **Don't hardcode model paths** - Use config, check models/ directory
6. **Don't ignore GPU OOM** - Catch CUDA errors, fallback to CPU
7. **Don't stream too fast** - Batch SSE updates to avoid flooding frontend

## Verification Checklist

Before committing, verify:
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `pytest tests/` passes
- [ ] `npm run check` passes
- [ ] New features have tests
- [ ] Sidecar imports work: `python -c "from backend.api.main import app"`

## Model Selection Quick Reference

| Task | Primary | Fallback | Notes |
|------|---------|----------|-------|
| Segmentation (text) | SAM3 | SAM2 | 4M+ concepts |
| Segmentation (point) | SAM2 | MobileSAM | 6x faster |
| Detection | YOLO11m | RT-DETR | 2.4ms inference |
| Open-vocab | YOLO-World | GroundingDINO | Text prompts |
| Faces | SCRFD-10G | YuNet | 95%+ accuracy |
| 3D Detection | PV-RCNN++ | CenterPoint | Point clouds |

## Available Slash Commands

- `/dev` - Start development environment
- `/test [rust|python|frontend|all]` - Run tests
- `/verify` - Full verification before commit
- `/commit-push-pr` - Commit, push, and create PR
- `/debug-sidecar` - Debug Python sidecar issues
- `/add-cv-model` - Add new CV model integration

## Available Subagents

- `cv-engineer` - CV model selection and integration
- `pointcloud-architect` - 3D data and 2D-3D fusion
- `verify-app` - End-to-end verification
- `code-simplifier` - Post-implementation cleanup
