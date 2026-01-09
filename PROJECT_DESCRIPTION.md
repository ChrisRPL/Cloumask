# Cloumask: Local-First Agentic CV Data Processing Platform

## Specification Document v1.0
**Last Updated:** December 31, 2025
**Author:** Krzysztof (CTO/Co-founder, Devrio)
**Project Name:** Cloumask
**Tagline:** *From cloud to canvas.*

---

## 1. Executive Summary

### 1.1 Vision
Cloumask is a local-first, agentic desktop application for computer vision data processing. It replaces complex CLI tools, fragmented scripts, and cloud-dependent platforms (Roboflow, Label Studio, CVAT) with a conversational AI interface that understands natural language commands like:

> "Take my dashcam footage in /data/drive_001, anonymize all faces and plates, then label vehicles and pedestrians, export to YOLO format"

### 1.2 Key Differentiators
- **Conversational-first UX** — Chat with your data pipeline, not config files
- **Human-in-the-loop execution** — Checkpoints, live previews, course correction
- **Local & private** — All processing on your machine, no cloud dependency
- **Unified 2D + 3D** — Images, videos, AND point clouds in one tool
- **Modern CV models** — SAM3, YOLO11/12, Florence-2, GroundingDINO running locally

### 1.3 Target Users
1. **ML/Robotics teams** at research institutes (primary — your institute)
2. **Solo researchers and developers** building CV datasets
3. **Labeling companies** needing efficient preprocessing and QA
4. **Enterprises** with privacy/compliance requirements (AV, healthcare, defense)

### 1.4 Business Model
**Open Core:**
- Core tool: Open source (MIT/Apache 2.0)
- Pro features: Collaboration, team management, priority support
- Enterprise: SSO, audit logs, custom model integration, on-premise deployment

---

## 2. Product Requirements

### 2.1 Core User Flows

#### Flow 1: Conversational Pipeline Creation
```
User: "I have ROS bags from our test drives in /data/rosbag_2025_12. 
       Extract the camera and LiDAR data, anonymize faces, 
       label all vehicles in 3D, and export to KITTI format."

Agent: "I found 47 ROS bag files (~2.3TB). Let me analyze the topics...

        Camera topics: /camera/front/image_raw (1920x1080, 30fps)
        LiDAR topics: /velodyne_points (VLP-64)
        
        Questions:
        1. Should I sync camera and LiDAR timestamps?
        2. For 3D vehicle labels, split into car/truck/bus or single 'vehicle' class?
        3. Anonymization: blur or black-box for faces?"

User: "Sync timestamps, split vehicle classes, black-box faces"

Agent: "Here's the pipeline:
        
        | Step | Action | Est. Time |
        |------|--------|-----------|
        | 1 | Extract synced frames | ~45 min |
        | 2 | Anonymize faces (SAM3 + detection) | ~2h |
        | 3 | 3D vehicle detection (PV-RCNN++) | ~3h |
        | 4 | Project 3D labels to 2D | ~30 min |
        | 5 | Export to KITTI | ~15 min |
        
        Checkpoints at: Step 2 (10%), Step 3 (25%)
        
        [Start Pipeline] [Edit Plan] [Save as Template]"
```

#### Flow 2: Human-in-the-Loop Execution
```
[Pipeline running - Step 2: Anonymization]

Progress: ████████░░░░░░░░░░░░ 42% | 1,247/2,970 frames

Agent: "⚠️ Checkpoint: I noticed 12 potential false positives on 
        billboards with faces. Want me to:
        
        1. [Ignore] - They'll be anonymized anyway
        2. [Add exclusion zone] - Skip detections > 2m² area
        3. [Show examples] - Review before deciding"

[Live preview: 6 processed frames with detection overlays]

Stats:
- Faces detected: 3,421
- Plates detected: 1,892  
- Avg confidence: 0.91
- Flagged for review: 12
```

#### Flow 3: Review Queue
```
Review Queue: 147 items

Sorted by:
├── Low confidence (89) - Most likely need correction
├── Anomalies (12) - Unusual detection patterns
├── Edge cases (46) - Occlusion, lighting issues

[Full-screen annotation editor]
- Accept (A) / Reject (R) / Edit (E) with keyboard
- "Apply same action to all similar" for batch operations
- Progress: 23/147 reviewed
```

### 2.2 Supported Data Types

#### 2D Data
| Format | Read | Write | Notes |
|--------|------|-------|-------|
| JPEG/PNG/WebP | ✓ | ✓ | Standard images |
| MP4/AVI/MKV | ✓ | - | Video frame extraction |
| TIFF (16-bit) | ✓ | ✓ | Medical/scientific imaging |

#### 3D Point Cloud Data
| Format | Read | Write | Notes |
|--------|------|-------|-------|
| PCD | ✓ | ✓ | Point Cloud Data (ASCII/binary) |
| PLY | ✓ | ✓ | Polygon File Format |
| LAS/LAZ | ✓ | ✓ | LiDAR standard, compressed |
| ROS bags | ✓ | - | Extract PointCloud2 topics |
| E57 | ✓ | ✓ | ASTM scanner format |
| nuScenes | ✓ | ✓ | Autonomous driving format |

#### Label Formats
| Format | Import | Export |
|--------|--------|--------|
| YOLO (v5/v8/v11) | ✓ | ✓ |
| COCO JSON | ✓ | ✓ |
| KITTI | ✓ | ✓ |
| Pascal VOC | ✓ | ✓ |
| CVAT XML | ✓ | ✓ |
| nuScenes | ✓ | ✓ |
| OpenLABEL | ✓ | ✓ |

### 2.3 Core Features

#### Anonymization
- **Face detection & anonymization** — SCRFD-10G or YuNet, blur/blackbox/pixelate
- **License plate detection** — YOLOv8 fine-tuned, region-aware
- **Text/signage blurring** — Scene text detection (PaddleOCR)
- **Point cloud anonymization** — Remove points near detected faces/plates

#### Auto-Labeling
- **Open-vocabulary detection** — SAM3 with text prompts ("all red cars")
- **Standard detection** — YOLO11/12, RT-DETR
- **Segmentation** — SAM3, SAM2, MobileSAM
- **3D object detection** — PV-RCNN++, CenterPoint, PointPillars

#### Dataset Operations
- **Duplicate detection** — Perceptual hashing, embedding similarity
- **Outlier detection** — Anomalous images/labels
- **Label QA** — Missing labels, overlapping boxes, out-of-bounds
- **Format conversion** — Any-to-any label format
- **Dataset splitting** — Train/val/test with stratification
- **Augmentation** — Albumentations integration

#### 2D-3D Fusion
- **Camera-LiDAR calibration** — Intrinsic/extrinsic matrix handling
- **3D → 2D projection** — Project 3D boxes onto camera images
- **2D → 3D lifting** — Estimate 3D from 2D labels + depth
- **Multi-sensor synchronization** — Timestamp alignment

---

## 3. Technical Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tauri 2.0 Shell                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Frontend (Svelte)                       │  │
│  │  ┌─────────┐ ┌─────────────┐ ┌─────────────────────────┐  │  │
│  │  │  Chat   │ │   Plan      │ │    Execution View       │  │  │
│  │  │  Panel  │ │   Editor    │ │  (Live Preview + Stats) │  │  │
│  │  └─────────┘ └─────────────┘ └─────────────────────────┘  │  │
│  │  ┌─────────────────────┐ ┌───────────────────────────────┐│  │
│  │  │  Point Cloud Viewer │ │      Review/Annotation UI     ││  │
│  │  │    (Three.js/Potree)│ │         (Canvas-based)        ││  │
│  │  └─────────────────────┘ └───────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │ IPC (Tauri Commands)              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Rust Core                               │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │  │
│  │  │  File I/O   │ │  Point Cloud│ │   Sidecar Manager   │  │  │
│  │  │  (pasture)  │ │  Processing │ │  (spawn/kill/stream)│  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/SSE/stdio
┌─────────────────────────────────────────────────────────────────┐
│                  Python Sidecar (FastAPI + PyInstaller)         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Agent Brain (LangGraph)                 │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │  │
│  │  │  Planner    │ │  Executor   │ │  Checkpoint Manager │  │  │
│  │  │  (LLM)      │ │  (Tools)    │ │  (State Persistence)│  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │                       CV Tools                             │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │  │
│  │  │ SAM3    │ │ YOLO12  │ │Florence2│ │  OpenPCDet      │  │  │
│  │  │ SAM2    │ │ YOLO11  │ │ GDINO   │ │  (3D detection) │  │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              Open3D / PDAL (Point Cloud)             │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Local LLM (Ollama)                           │
│                    Qwen3-14B / Llama-3.1-8B                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Technology Stack

#### Desktop Shell
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | **Tauri 2.0** | 600KB-10MB apps vs Electron's 100MB+, Rust backend |
| Frontend | **Svelte 5** | Compiled output, no runtime, excellent performance |
| UI Components | **shadcn/ui + Tailwind** | Beautifully designed, accessible |
| 3D Visualization | **Three.js + Potree-core** | Handles millions of points with LOD |

#### Agent & LLM
| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM Inference | **Ollama** | Ease of use, OpenAI-compatible API, tool calling |
| LLM Models | **Qwen3-14B** (primary), **Llama-3.1-8B** (fallback) | 0.971 F1 tool-calling, runs on RTX 3080 |
| Agent Framework | **LangGraph 1.0.5** | Best checkpoint + human-in-the-loop support (Dec 2025) |
| Agent Pattern | **Single agent with tools** | Sequential pipeline, no multi-agent overhead |

> **Note on LangGraph (Dec 2025):** LangGraph 1.0 was released Oct 22, 2025 as the first stable major version. Current version 1.0.5 (Dec 12, 2025) includes enhanced `create_agent`, model retry middleware, and content moderation middleware. The `langgraph.prebuilt` module is deprecated in favor of `langchain.agents`.

#### Computer Vision Models
| Task | Primary Model | Fallback | Notes |
|------|---------------|----------|-------|
| Segmentation (text prompts) | **SAM3** | SAM2 | NEW! Nov 2025, 4M+ concepts |
| Segmentation (point prompts) | **SAM2** | MobileSAM | 6x faster than SAM1, video native |
| Object Detection | **YOLO11m** | YOLOv12n | 2.4ms inference, best speed/accuracy |
| Open-Vocab Detection | **YOLO-World** | GroundingDINO | 50+ FPS, text prompts |
| Multi-task | **Florence-2** | - | Detection + segmentation + captioning |
| Face Detection | **SCRFD-10G** | YuNet | 95%+ WIDER FACE, or 1.6ms real-time |
| 3D Detection | **PV-RCNN++** | CenterPoint | 84% 3D AP on KITTI |

> **Note on YOLO versions (Dec 2025):**
> - **YOLO11** (Oct 2024): Current stable, recommended for production
> - **YOLOv12** (Feb 2025): Attention-based, slightly slower but more accurate
> - **YOLO26** (announced Sept 2025): **NOT YET RELEASED** — NMS-free, 43% faster CPU inference. Monitor for release in early 2026.

#### Point Cloud Processing
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Python Library | **Open3D** | Comprehensive, 11K GitHub stars |
| Rust Library | **pasture** | LAS/LAZ/3D Tiles, AoS/SoA memory |
| Pipeline Tool | **PDAL** | 30+ format drivers |
| 3D ML Framework | **OpenPCDet** | Unified framework, pre-trained weights |
| ROS Bag Parsing | **rosbag + ros_numpy** | Native ROS approach |

#### Python Backend
| Component | Choice | Rationale |
|-----------|--------|-----------|
| API Framework | **FastAPI** | Async, SSE streaming, type hints |
| Bundling | **PyInstaller** | Single executable sidecar |
| GPU Acceleration | **PyTorch + CUDA 12.x** | Native GPU support |

### 3.3 Agent Architecture

#### Single Agent with Tool Calling
```python
# LangGraph state machine
class PipelineState(TypedDict):
    messages: list[Message]
    plan: list[PipelineStep]
    current_step: int
    execution_results: dict
    checkpoints: list[Checkpoint]
    user_feedback: Optional[str]

# Graph definition
graph = StateGraph(PipelineState)
graph.add_node("understand", understand_request)
graph.add_node("plan", create_plan)
graph.add_node("await_approval", human_approval_checkpoint)
graph.add_node("execute_step", execute_current_step)
graph.add_node("checkpoint", checkpoint_handler)
graph.add_node("complete", finalize_pipeline)

# Edges with conditions
graph.add_edge("understand", "plan")
graph.add_edge("plan", "await_approval")
graph.add_conditional_edges("await_approval", check_approval, {
    "approved": "execute_step",
    "edit": "plan",
    "cancel": END
})
graph.add_conditional_edges("execute_step", check_next_action, {
    "checkpoint": "checkpoint",
    "next_step": "execute_step",
    "complete": "complete"
})
graph.add_edge("checkpoint", "await_approval")
```

#### Tool Definitions
```python
TOOLS = [
    # Data Ingestion
    Tool("scan_directory", scan_directory, "Analyze folder contents"),
    Tool("extract_frames", extract_frames, "Extract frames from video"),
    Tool("parse_rosbag", parse_rosbag, "Extract data from ROS bags"),
    
    # Anonymization
    Tool("anonymize", anonymize, "Anonymize faces, plates, text"),
    Tool("anonymize_pointcloud", anonymize_pointcloud, "Remove points near faces/plates"),
    
    # Auto-Labeling
    Tool("detect_objects", detect_objects, "YOLO/RT-DETR detection"),
    Tool("segment_sam3", segment_sam3, "SAM3 text-prompted segmentation"),
    Tool("segment_sam2", segment_sam2, "SAM2 point-prompted segmentation"),
    Tool("detect_3d", detect_3d, "3D object detection on point clouds"),
    
    # Fusion
    Tool("project_3d_to_2d", project_3d_to_2d, "Project 3D labels to camera"),
    Tool("sync_sensors", sync_sensors, "Synchronize multi-sensor data"),
    
    # Dataset Operations
    Tool("find_duplicates", find_duplicates, "Find duplicate/similar images"),
    Tool("label_qa", label_qa, "Check label quality"),
    Tool("convert_format", convert_format, "Convert label formats"),
    Tool("split_dataset", split_dataset, "Create train/val/test splits"),
    Tool("export", export, "Export final dataset"),
    
    # Reporting
    Tool("generate_report", generate_report, "Generate HTML quality report"),
]
```

### 3.4 Checkpoint System

#### Trigger Conditions
```python
CHECKPOINT_TRIGGERS = {
    # Percentage-based (configurable per step)
    "percentage": [0.1, 0.25, 0.5],
    
    # Quality-based (auto-triggered)
    "confidence_drop": 0.15,      # Avg confidence drops 15%
    "error_rate": 0.05,           # >5% errors detected
    "anomaly_spike": 0.03,        # >3% anomalous results
    
    # Always checkpoint after critical steps
    "critical_steps": ["anonymize", "segment_sam3", "detect_3d"],
}
```

#### Checkpoint State Persistence
```python
@dataclass
class Checkpoint:
    step_id: int
    step_name: str
    progress: float
    stats: dict
    samples: list[str]  # Paths to sample outputs
    issues: list[Issue]
    timestamp: datetime
    
    def to_sqlite(self, conn):
        """Persist to SQLite for resume capability"""
        ...
```

---

## 4. UI/UX Design

### 4.1 Design Principles

1. **Conversation as primary input** — Chat is the main way to interact
2. **Progressive disclosure** — Simple by default, power features accessible
3. **Live feedback** — Always show what's happening, never leave user waiting
4. **Keyboard-first** — Power users can navigate without mouse
5. **Dark mode default** — Data tools are used in long sessions

### 4.2 Color Palette

```css
/* Dark theme (default) */
--bg-primary: #0a0a0b;      /* Main background */
--bg-secondary: #18181b;    /* Cards, panels */
--bg-tertiary: #27272a;     /* Hover states */
--border: #3f3f46;          /* Borders */

--text-primary: #fafafa;    /* Main text */
--text-secondary: #a1a1aa;  /* Secondary text */
--text-muted: #71717a;      /* Muted text */

--accent-primary: #8b5cf6;  /* Violet - primary actions */
--accent-success: #22c55e;  /* Green - success states */
--accent-warning: #f59e0b;  /* Amber - warnings/checkpoints */
--accent-error: #ef4444;    /* Red - errors */
```

### 4.3 Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌────────┐                                          [_][□][X] │
│  │ Logo   │  Cloumask                    [Project: berlin_av] │
├──┴────────┴─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌───────────────────────────────────────────┐│
│  │   Sidebar    │  │              Main Content                 ││
│  │              │  │                                           ││
│  │  • Chat      │  │  [Content changes based on sidebar tab]   ││
│  │  • Plan      │  │                                           ││
│  │  • Execution │  │                                           ││
│  │  • Review    │  │                                           ││
│  │  • Export    │  │                                           ││
│  │              │  │                                           ││
│  │  ─────────── │  │                                           ││
│  │  Project     │  │                                           ││
│  │  12,847 imgs │  │                                           ││
│  │              │  │                                           ││
│  │  ─────────── │  │                                           ││
│  │  Models      │  │                                           ││
│  │  SAM3    ●   │  │                                           ││
│  │  YOLO11  ●   │  │                                           ││
│  │  Qwen3   ●   │  │                                           ││
│  └──────────────┘  └───────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Key Views

#### Chat View
- Message bubbles with user/agent distinction
- Inline clarifying questions with quick-reply buttons
- Plan preview embedded in agent messages
- "Start Pipeline" / "Edit Plan" CTAs

#### Execution View
- **Left panel:** Pipeline steps with progress
- **Center:** Live preview grid (6 images, updating)
- **Top:** Checkpoint alert banner when triggered
- **Bottom:** Stats dashboard (processed, detected, flagged)
- **Right:** Agent commentary stream

#### Review View
- **Left:** Review queue with filtering
- **Center:** Full-screen annotation canvas
- **Bottom:** Accept/Reject/Edit controls + keyboard hints
- Batch operations for similar items

#### Point Cloud View
- Three.js viewer with orbit controls
- Color by: intensity, height, classification, RGB
- 3D bounding box visualization
- Synchronized 2D camera view panel

---

## 5. Implementation Plan

### Phase 1: Foundation (Weeks 1-3)

#### Week 1: Project Setup
- [ ] Initialize Tauri 2.0 + Svelte project
- [ ] Set up Python sidecar with FastAPI
- [ ] Configure PyInstaller bundling
- [ ] Basic IPC: frontend ↔ Rust ↔ Python
- [ ] Install Ollama, verify Qwen3-14B runs

#### Week 2: Agent Brain MVP
- [ ] Set up LangGraph with basic state machine
- [ ] Implement tool calling with Ollama
- [ ] Create 3 tools: `scan_directory`, `anonymize`, `export`
- [ ] Chat UI: messages, agent responses, streaming

#### Week 3: First Pipeline
- [ ] Conversational flow: request → clarify → plan
- [ ] Plan visualization component
- [ ] Basic execution with progress
- [ ] End-to-end: chat → anonymize folder → export

### Phase 2: Core Features (Weeks 4-7)

#### Week 4: Anonymization Complete
- [ ] Face detection with SCRFD
- [ ] License plate detection
- [ ] SAM3 integration for segmentation
- [ ] Blur/blackbox/pixelate options
- [ ] Live preview during processing

#### Week 5: Auto-Labeling
- [ ] YOLO11 integration
- [ ] SAM3 text-prompted segmentation
- [ ] GroundingDINO/YOLO-World for open-vocab
- [ ] Confidence thresholds, class mapping
- [ ] Label format export (YOLO, COCO, KITTI)

#### Week 6: Checkpoints & Review
- [ ] Checkpoint trigger system
- [ ] Checkpoint UI with action buttons
- [ ] Review queue implementation
- [ ] Annotation editor (bounding boxes)
- [ ] Keyboard shortcuts

#### Week 7: Dataset Operations
- [ ] Duplicate detection
- [ ] Label QA checks
- [ ] Format conversion
- [ ] Dataset splitting
- [ ] HTML report generation

### Phase 3: Point Cloud Support (Weeks 8-10)

#### Week 8: Point Cloud I/O
- [ ] PCD/PLY/LAS reading with Open3D
- [ ] ROS bag parsing
- [ ] Point cloud viewer (Three.js + Potree)
- [ ] Basic visualization controls

#### Week 9: 3D Processing
- [ ] Point cloud anonymization
- [ ] Integration with OpenPCDet
- [ ] 3D bounding box visualization
- [ ] 3D → 2D projection

#### Week 10: Multi-Sensor Fusion
- [ ] Camera-LiDAR calibration handling
- [ ] Timestamp synchronization
- [ ] Synchronized 2D/3D view
- [ ] nuScenes/KITTI format support

### Phase 4: Polish & Distribution (Weeks 11-12)

#### Week 11: UX Polish
- [ ] Loading states, error handling
- [ ] Keyboard navigation throughout
- [ ] Settings panel (model selection, GPU config)
- [ ] Template saving/loading

#### Week 12: Distribution
- [ ] Cross-platform testing (Linux, Windows, macOS)
- [ ] Installer packaging
- [ ] Documentation
- [ ] Landing page
- [ ] Beta user onboarding

---

## 6. File Structure

```
dataforge/
├── src-tauri/                 # Rust/Tauri backend
│   ├── src/
│   │   ├── main.rs            # Tauri entry point
│   │   ├── commands/          # IPC command handlers
│   │   ├── pointcloud/        # Rust point cloud processing
│   │   └── sidecar.rs         # Python sidecar management
│   ├── binaries/              # Bundled Python executable
│   ├── Cargo.toml
│   └── tauri.conf.json
│
├── src/                       # Svelte frontend
│   ├── lib/
│   │   ├── components/
│   │   │   ├── Chat/          # Chat UI components
│   │   │   ├── Plan/          # Plan editor
│   │   │   ├── Execution/     # Live execution view
│   │   │   ├── Review/        # Review queue & editor
│   │   │   ├── PointCloud/    # 3D viewer
│   │   │   └── ui/            # shadcn components
│   │   ├── stores/            # Svelte stores
│   │   └── utils/
│   ├── routes/
│   └── app.html
│
├── backend/                   # Python sidecar
│   ├── api/
│   │   ├── main.py            # FastAPI app
│   │   ├── routes/            # API endpoints
│   │   └── streaming.py       # SSE handlers
│   ├── agent/
│   │   ├── graph.py           # LangGraph definition
│   │   ├── tools/             # Tool implementations
│   │   ├── prompts/           # System prompts
│   │   └── checkpoints.py     # Checkpoint logic
│   ├── cv/
│   │   ├── detection.py       # YOLO, RT-DETR
│   │   ├── segmentation.py    # SAM2, SAM3
│   │   ├── anonymization.py   # Face/plate blurring
│   │   └── pointcloud.py      # 3D processing
│   ├── data/
│   │   ├── loaders.py         # Format readers
│   │   ├── exporters.py       # Format writers
│   │   └── transforms.py      # Augmentations
│   ├── requirements.txt
│   └── pyproject.toml
│
├── models/                    # Local model weights
│   ├── sam3/
│   ├── yolo11/
│   └── scrfd/
│
├── docs/
├── tests/
└── README.md
```

---

## 7. Dependencies

### Python (backend/requirements.txt)
```
# API
fastapi>=0.115.0
uvicorn>=0.32.0
sse-starlette>=2.1.0

# Agent
langchain>=0.3.0
langgraph>=0.2.0
langchain-ollama>=0.2.0

# CV Models
torch>=2.5.0
torchvision>=0.20.0
ultralytics>=8.3.0  # YOLO11, YOLO12
segment-anything-2>=1.0.0
# sam3 - install from git when available
transformers>=4.46.0
groundingdino>=0.1.0

# Point Cloud
open3d>=0.18.0
laspy>=2.5.0
rosbag  # For ROS bag parsing

# Image Processing
opencv-python>=4.10.0
pillow>=11.0.0
albumentations>=1.4.0

# Utils
numpy>=2.0.0
pandas>=2.2.0
tqdm>=4.66.0
```

### Rust (src-tauri/Cargo.toml)
```toml
[dependencies]
tauri = { version = "2", features = ["shell-sidecar"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
pasture-core = "0.4"  # Point cloud
pasture-io = "0.4"
notify = "6"  # File watching
```

### Frontend (package.json)
```json
{
  "dependencies": {
    "@tauri-apps/api": "^2",
    "svelte": "^5",
    "three": "^0.170",
    "@pnext/three-loader": "^1.3",
    "bits-ui": "^1.0",
    "tailwindcss": "^3.4",
    "lucide-svelte": "^0.454"
  }
}
```

---

## 8. Model Downloads & GPU Requirements

### Minimum Hardware
- **GPU:** NVIDIA RTX 3070 (8GB VRAM) or equivalent
- **RAM:** 32GB
- **Storage:** 50GB for models + working space

### Recommended Hardware
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **RAM:** 64GB
- **Storage:** 500GB SSD

### Model Sizes (approximate)
| Model | Size | VRAM @ Inference |
|-------|------|------------------|
| Qwen3-14B (Q4) | ~8GB | ~10GB |
| SAM3 | ~3.5GB | ~8GB |
| SAM2-Large | ~2.4GB | ~6GB |
| YOLO11m | ~40MB | ~2GB |
| SCRFD-10G | ~16MB | ~1GB |
| PV-RCNN++ | ~150MB | ~4GB |

### First-Run Model Download
```
On first launch:
  → Check for Ollama installation
  → Pull Qwen3-14B (or user-selected model)
  → Download SAM3 weights from HuggingFace
  → Download YOLO11 weights
  → Verify GPU detection
```

---

## 9. Success Metrics

### MVP Success (Week 12)
- [ ] Complete anonymization + labeling pipeline works end-to-end
- [ ] Processing speed: 10+ images/second for detection
- [ ] 3 internal users at institute using it weekly
- [ ] Cross-platform builds for Linux, Windows, macOS

### 3-Month Goals
- [ ] 100+ GitHub stars
- [ ] 10 external beta users
- [ ] Point cloud pipeline fully functional
- [ ] Featured in 1 ML community (Reddit, HN, Twitter)

### 6-Month Goals
- [ ] 1,000+ downloads
- [ ] First paying customer (Pro tier)
- [ ] 5+ label format support
- [ ] Community contributions (PRs, issues)

---

## 10. Open Questions

1. ~~**Name confirmation:**~~ ✅ **Cloumask** (cloud + mask = point cloud + segmentation)
2. **License:** Apache 2.0 or MIT for open source core?
3. **SAM3 access:** Currently requires HuggingFace approval — fallback to SAM2?
4. **Cloud option:** Completely out of scope, or optional for heavy models?
5. **Multi-user:** SQLite + file locks sufficient, or need proper backend?

> **Note on Cloud LLMs (if added later, Dec 2025 landscape):**
> - **Claude Opus 4.5** (Nov 24, 2025): SOTA for coding, 80.9% SWE-bench Verified
> - **GPT-5.2** (Dec 11, 2025): Latest OpenAI, strong reasoning
> - **Gemini 3** (Nov 18, 2025): 1M token context, 1501 Elo on LMArena
> - For a cloud-optional feature, Claude Opus 4.5 would be ideal for complex agentic planning.

---

## Appendix A: Agent System Prompt

```markdown
You are Cloumask, an AI assistant specialized in computer vision data processing. 
Your role is to help users prepare datasets for ML training through conversational interaction.

## Capabilities
You can: scan directories, extract video frames, parse ROS bags, anonymize faces/plates, 
auto-label objects (2D and 3D), segment images, convert label formats, create dataset splits,
find duplicates, check label quality, and export final datasets.

## Interaction Style
1. When a user describes a task, first scan their data to understand what you're working with
2. Ask 2-4 clarifying questions to understand requirements (not more)
3. Propose a concrete plan with estimated times
4. Wait for approval before executing
5. During execution, proactively surface issues and ask for guidance

## Tool Usage
- Always use `scan_directory` first to understand the data
- For anonymization, prefer SAM3 for quality, MobileSAM for speed
- For detection, use YOLO11 unless open-vocabulary is needed (then YOLO-World)
- For 3D data, always check sensor calibration before projection

## Checkpoint Behavior
- Pause at 10% of anonymization to verify quality
- Pause at 25% of labeling to check detections
- Alert immediately if confidence drops below threshold
- Show sample previews at every checkpoint

## Response Format
- Keep responses concise and actionable
- Use tables for plans and comparisons
- Include time estimates
- Offer clear action buttons: [Start] [Edit] [Cancel]
```

---

## Appendix B: Competitive Analysis Summary

| Feature | Cloumask | Label Studio | CVAT | Roboflow |
|---------|-----------|--------------|------|----------|
| Local-first | ✓ | Partial | ✓ | ✗ |
| Conversational UI | ✓ | ✗ | ✗ | ✗ |
| Human-in-the-loop | ✓ | ✗ | ✗ | ✗ |
| Point cloud support | ✓ | ✗ | ✓ | ✗ |
| 2D-3D fusion | ✓ | ✗ | ✗ | ✗ |
| SAM3 integration | ✓ | ✗ | ✗ | ✓ |
| Auto-labeling | ✓ | Plugin | Plugin | ✓ |
| Open source | ✓ | ✓ | ✓ | ✗ |
| Free tier | ✓ | ✓ | ✓ | Limited |

---

*End of Specification Document*
