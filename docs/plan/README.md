# Cloumask Development Plan

> **Status:** 🟡 In Progress (Phase 1-4 Complete)
> **Last Updated:** January 2026

*From cloud to canvas — Local-first agentic CV data processing*

---

## Vision

Cloumask replaces complex CLI tools, fragmented scripts, and cloud-dependent platforms with a conversational AI interface that understands natural language commands like:

> "Take my dashcam footage in /data/drive_001, anonymize all faces and plates, then label vehicles and pedestrians, export to YOLO format"

---

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

---

## Module Status

| Module | Status | Priority | Spec |
|--------|--------|----------|------|
| **Foundation** | 🟢 Complete | P0 | [SPEC](./01-foundation/SPEC.md) |
| **Agent System** | 🟢 Complete | P0 | [SPEC](./02-agent-system/SPEC.md) |
| **CV Models** | 🟢 Complete | P0 | [SPEC](./03-cv-models/SPEC.md) |
| **Frontend UI** | 🟢 Complete | P1 | [SPEC](./04-frontend-ui/SPEC.md) |
| **Point Cloud** | 🟢 Complete | P1 | [SPEC](./05-point-cloud/SPEC.md) |
| **Data Pipeline** | 🔴 Not Started | P1 | [SPEC](./06-data-pipeline/SPEC.md) |

---

## Development Phases

### Phase 1: Foundation
- [ ] Initialize Tauri 2.0 + Svelte 5 project
- [ ] Set up Python sidecar with FastAPI
- [ ] Configure PyInstaller bundling
- [ ] Basic IPC: Frontend ↔ Rust ↔ Python
- [ ] Verify Ollama + Qwen3-14B

### Phase 2: Agent Brain MVP
- [ ] Set up LangGraph state machine
- [ ] Implement tool calling with Ollama
- [ ] Create initial tools: `scan_directory`, `anonymize`, `export`
- [ ] Chat UI with streaming responses

### Phase 3: Core CV Features
- [ ] Face detection + anonymization (SCRFD)
- [ ] Object detection (YOLO11)
- [ ] Segmentation (SAM3)
- [ ] Checkpoint system
- [ ] Review queue

### Phase 4: Point Cloud Support
- [ ] Point cloud I/O (pasture + Open3D)
- [ ] 3D viewer (Three.js)
- [ ] 3D object detection (PV-RCNN++)
- [ ] 2D-3D fusion

### Phase 5: Polish & Distribution
- [ ] Cross-platform builds
- [ ] Installer packaging
- [ ] Documentation

---

## Model Selection (January 2026)

| Task | Primary | Fallback | Notes |
|------|---------|----------|-------|
| Detection | YOLO11m | YOLO26, RF-DETR | 2.4ms inference |
| Segmentation | SAM3 | SAM2 | Text prompts, 4M+ concepts |
| Open-Vocab | YOLO-World | GroundingDINO | 50+ FPS |
| Faces | SCRFD-10G | YuNet | 95%+ accuracy |
| 3D Detection | PV-RCNN++ | CenterPoint, BEVFusion | 84% 3D AP |
| Local LLM | Qwen3-14B | Llama 4, Qwen3-8B | Best tool calling |

---

## Quick Links

- [Project Description](../../PROJECT_DESCRIPTION.md) - Full specification
- [CLAUDE.md](../../CLAUDE.md) - Claude Code guidance
- [.claude/](../../.claude/) - Claude Code configuration

---

## How to Update This Plan

1. Update module status in the table above
2. Check off completed tasks in phase sections
3. Add notes to individual SPEC.md files
4. Create sub-specs in module folders as features are implemented
