# Cloumask

> *From cloud to canvas — Local-first agentic CV data processing*

Cloumask is a conversational AI desktop application for computer vision data processing. It replaces complex CLI tools, fragmented scripts, and cloud-dependent platforms with natural language commands like:

> "Take my dashcam footage in /data/drive_001, anonymize all faces and plates, then label vehicles and pedestrians, export to YOLO format"

## Key Features

- **Conversational-first UX** — Chat with your data pipeline, not config files
- **Human-in-the-loop execution** — Checkpoints, live previews, course correction
- **Local & private** — All processing on your machine, no cloud dependency
- **Unified 2D + 3D** — Images, videos, AND point clouds in one tool
- **Modern CV models** — SAM3, YOLO11, Florence-2, GroundingDINO running locally

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tauri 2.0 Shell                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Frontend (Svelte 5)                     │  │
│  │  ┌─────────┐ ┌─────────────┐ ┌─────────────────────────┐  │  │
│  │  │  Chat   │ │    Plan     │ │    Execution View       │  │  │
│  │  │  Panel  │ │   Editor    │ │  (Live Preview + Stats) │  │  │
│  │  └─────────┘ └─────────────┘ └─────────────────────────┘  │  │
│  │  ┌─────────────────────┐ ┌───────────────────────────────┐│  │
│  │  │  Point Cloud Viewer │ │      Review/Annotation UI     ││  │
│  │  │    (Three.js)       │ │         (Canvas-based)        ││  │
│  │  └─────────────────────┘ └───────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │ IPC                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Rust Core                               │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │  │
│  │  │  File I/O   │ │  Point Cloud│ │   Sidecar Manager   │  │  │
│  │  │  (pasture)  │ │  Processing │ │  (spawn/kill/stream)│  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/SSE (port 8765)
┌─────────────────────────────────────────────────────────────────┐
│                  Python Sidecar (FastAPI)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Agent Brain (LangGraph)                 │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │  │
│  │  │  Planner    │ │  Executor   │ │  Checkpoint Manager │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │                       CV Models                            │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │  │
│  │  │  SAM3   │ │ YOLO11  │ │ SCRFD   │ │    PV-RCNN++    │  │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Local LLM (Ollama)                           │
│                    Qwen3-14B / Llama 4                          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- **Node.js 20+** and npm
- **Rust 1.75+** with cargo
- **Python 3.11+**
- [Ollama](https://ollama.ai) (optional, for LLM features)

### Setup

```bash
# Install frontend dependencies
npm install

# Install Python sidecar dependencies
cd backend && pip install -r requirements.txt && cd ..

# Start development (launches Tauri + Python sidecar)
cargo tauri dev
```

### Verify Installation

- Open app → Status dashboard shows all systems green
- Python sidecar health: http://localhost:8765/health
- API docs: http://localhost:8765/docs

## Current Features (v0.1.0)

- Tauri 2.0 desktop shell with Svelte 5 frontend
- Python FastAPI sidecar with auto-start/stop lifecycle
- Real-time health monitoring dashboard
- Ollama LLM integration (status, models, generation)
- Type-safe IPC across all layers (Frontend ↔ Rust ↔ Python)

## Tech Stack

| Layer | Technology |
|-------|------------|
| Desktop Shell | Tauri 2.0 (Rust) |
| Frontend | Svelte 5 + shadcn/ui + Tailwind |
| 3D Visualization | Three.js |
| Agent Framework | LangGraph |
| Local LLM | Ollama (Qwen3-14B) |
| CV Models | SAM3, YOLO11, SCRFD, PV-RCNN++ |
| Point Cloud | pasture (Rust), Open3D (Python) |

## Supported Data

### Formats

| Type | Formats |
|------|---------|
| Images | JPEG, PNG, WebP, TIFF |
| Video | MP4, AVI, MKV |
| Point Cloud | PCD, PLY, LAS/LAZ, E57, ROS bags |
| Labels | YOLO, COCO, KITTI, Pascal VOC, nuScenes |

### CV Capabilities

| Task | Primary Model | Notes |
|------|---------------|-------|
| Segmentation | SAM3 | Text prompts, 4M+ concepts |
| Detection | YOLO11m | 2.4ms inference |
| Open-Vocab | YOLO-World | 50+ FPS, text prompts |
| Faces | SCRFD-10G | 95%+ accuracy |
| 3D Detection | PV-RCNN++ | 84% 3D AP on KITTI |

## Development Status

See [docs/plan/](docs/plan/) for detailed implementation plans.

| Module | Status |
|--------|--------|
| Foundation | 🟢 Complete |
| Agent System | 🔴 Not Started |
| CV Models | 🔴 Not Started |
| Frontend UI | 🔴 Not Started |
| Point Cloud | 🔴 Not Started |
| Data Pipeline | 🔴 Not Started |

## Requirements

### Minimum Hardware
- GPU: NVIDIA RTX 3070 (8GB VRAM)
- RAM: 32GB
- Storage: 50GB

### Recommended Hardware
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- RAM: 64GB
- Storage: 500GB SSD

## Development

```bash
# Start development
cargo tauri dev                           # Full app with hot reload
npm run dev                               # Frontend only
cd backend && python -m uvicorn backend.api.main:app --reload --port 8765  # Python sidecar

# Testing
cd src-tauri && cargo test               # Rust tests (8 tests)
cd backend && PYTHONPATH=src pytest tests/ -v  # Python tests (22 tests)
npm run check                            # Frontend type checking

# Build
cargo tauri build                        # Production build
```

## Documentation

- [Project Description](PROJECT_DESCRIPTION.md) — Full specification
- [Development Plan](docs/plan/) — Implementation roadmap
- [CLAUDE.md](CLAUDE.md) — Claude Code guidance

## License

TBD (Apache 2.0 or MIT)
