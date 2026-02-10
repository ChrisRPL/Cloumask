# Frontend UI Module

> **Status:** 🟢 Complete (Implemented; release QA backlog remains)
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation, 02-agent-system

## Overview

Build the Svelte 5 frontend with five main views: Chat, Plan Editor, Execution View, Review Queue, and Point Cloud Viewer. Keyboard-first navigation with brand-aligned theming.

## Goals

- [x] Conversational chat interface with streaming responses
- [x] Visual pipeline plan editor
- [x] Live execution view with preview grid
- [x] Review queue for annotation corrections
- [x] Three.js point cloud viewer
- [x] Responsive brand-aligned UI

## Technical Design

### Design System
- **Framework:** Svelte 5 with runes ($state, $derived, $effect)
- **Components:** shadcn/ui (bits-ui + Tailwind)
- **Icons:** Lucide
- **Theme:** Forest/cream palette with monospace-first typography

### Layout Structure
```
┌─────────────────────────────────────────────────────────────────┐
│  ┌────────┐                                          [_][□][X] │
│  │ Logo   │  Cloumask                    [Project: project_1]  │
├──┴────────┴─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌───────────────────────────────────────────┐│
│  │   Sidebar    │  │              Main Content                 ││
│  │              │  │                                           ││
│  │  • Chat      │  │  [Changes based on sidebar selection]     ││
│  │  • Plan      │  │                                           ││
│  │  • Execute   │  │                                           ││
│  │  • Review    │  │                                           ││
│  │  • Export    │  │                                           ││
│  └──────────────┘  └───────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### State Management
- **Svelte stores** for global state (agent, pipeline, settings)
- **Runes** for component-local state
- **SSE subscription** in dedicated store

## Implementation Tasks

- [x] **Core Layout**
  - [x] App shell with sidebar navigation
  - [x] Brand theme configuration
  - [x] Responsive breakpoints
  - [x] Keyboard navigation (vim-like)

- [x] **Chat Panel**
  - [x] Message list with user/agent distinction
  - [x] Streaming text display
  - [x] Inline clarification questions
  - [x] Quick-reply buttons
  - [x] Plan preview in messages

- [x] **Plan Editor**
  - [x] Pipeline step visualization
  - [x] Drag-and-drop reordering
  - [x] Step configuration panels
  - [x] Start/Edit/Cancel CTAs

- [x] **Execution View**
  - [x] Progress bar with step indicators
  - [x] Live preview grid (6 images)
  - [x] Checkpoint alert banner
  - [x] Stats dashboard (processed, detected, flagged)
  - [x] Agent commentary stream

- [x] **Review Queue**
  - [x] Filterable item list
  - [x] Full-screen annotation canvas
  - [x] Bounding box editing
  - [x] Accept/Reject/Edit controls
  - [x] Keyboard shortcuts (A/R/E)

- [x] **Stores & State**
  - [x] `agent.ts` - Agent state, messages
  - [x] `pipeline.ts` - Current pipeline state
  - [x] `settings.ts` - User preferences
  - [x] `sse.ts` - SSE connection management

## Acceptance Criteria

- [x] Chat shows streaming responses from agent
- [x] Plan editor displays pipeline steps visually
- [x] Execution view updates in real-time during processing
- [x] Review queue allows editing bounding boxes
- [x] All views navigable via keyboard
- [x] Theme tokens render correctly on all views

## Files to Create/Modify

```
src/
├── routes/
│   ├── +layout.svelte      # App shell
│   ├── +page.svelte        # Main view router
│   └── +error.svelte       # Error page
├── lib/
│   ├── components/
│   │   ├── Chat/
│   │   │   ├── ChatPanel.svelte
│   │   │   ├── MessageList.svelte
│   │   │   ├── MessageBubble.svelte
│   │   │   └── QuickReply.svelte
│   │   ├── Plan/
│   │   │   ├── PlanEditor.svelte
│   │   │   ├── PipelineStep.svelte
│   │   │   └── StepConfig.svelte
│   │   ├── Execution/
│   │   │   ├── ExecutionView.svelte
│   │   │   ├── PreviewGrid.svelte
│   │   │   ├── ProgressBar.svelte
│   │   │   └── StatsPanel.svelte
│   │   ├── Review/
│   │   │   ├── ReviewQueue.svelte
│   │   │   ├── AnnotationCanvas.svelte
│   │   │   └── ReviewControls.svelte
│   │   ├── Sidebar/
│   │   │   └── Sidebar.svelte
│   │   └── ui/             # shadcn components
│   ├── stores/
│   │   ├── agent.ts
│   │   ├── pipeline.ts
│   │   ├── settings.ts
│   │   └── sse.ts
│   └── utils/
│       ├── tauri.ts        # IPC helpers
│       └── keyboard.ts     # Keyboard navigation
```

## Sub-Specs (Detailed Implementation)

The frontend UI module is broken down into 10 atomic task specifications. Complete these in order (critical path):

### Foundation Layer (P0 - Critical)
| # | Spec | Description | Complexity |
|---|------|-------------|------------|
| 01 | [01-design-system.md](01-design-system.md) | Tailwind, shadcn/ui, theme tokens, base components | Medium |
| 02 | [02-core-layout.md](02-core-layout.md) | App shell, sidebar, header, view routing | Medium |
| 03 | [03-stores-state.md](03-stores-state.md) | Svelte stores for agent, pipeline, execution, review | High |
| 04 | [04-tauri-sse-integration.md](04-tauri-sse-integration.md) | Tauri IPC wrappers, SSE connection manager | High |

### View Components (P1 - High)
| # | Spec | Description | Complexity |
|---|------|-------------|------------|
| 05 | [05-chat-panel.md](05-chat-panel.md) | Chat interface, streaming, clarifications, quick replies | High |
| 06 | [06-plan-editor.md](06-plan-editor.md) | Pipeline visualization, drag-drop, step config | High |
| 07 | [07-execution-view.md](07-execution-view.md) | Progress, preview grid, checkpoints, stats | High |
| 08 | [08-review-queue.md](08-review-queue.md) | Annotation canvas, bbox editing, review controls | Very High |

### Advanced Features (P2 - Medium)
| # | Spec | Description | Complexity |
|---|------|-------------|------------|
| 09 | [09-pointcloud-viewer.md](09-pointcloud-viewer.md) | Three.js viewer, 3D navigation, point rendering | Very High |
| 10 | [10-keyboard-navigation.md](10-keyboard-navigation.md) | Global shortcuts, vim-like nav, command palette | Medium |

### Implementation Order

```
01-design-system ─┬─▶ 02-core-layout ─┬─▶ 05-chat-panel
                  │                    │
                  ├─▶ 03-stores-state ─┼─▶ 06-plan-editor
                  │                    │
                  └─▶ 04-tauri-sse ────┼─▶ 07-execution-view
                                       │
                                       ├─▶ 08-review-queue
                                       │
                                       ├─▶ 09-pointcloud-viewer
                                       │
                                       └─▶ 10-keyboard-navigation
```

### Progress Tracking

- [x] 01-design-system - Tailwind + shadcn/ui configured
- [x] 02-core-layout - App shell functional
- [x] 03-stores-state - All stores implemented
- [x] 04-tauri-sse-integration - Backend communication working
- [x] 05-chat-panel - Chat with streaming working
- [x] 06-plan-editor - Plan editing functional
- [x] 07-execution-view - Live progress updates
- [x] 08-review-queue - Annotation editing works
- [x] 09-pointcloud-viewer - 3D viewer renders
- [x] 10-keyboard-navigation - All shortcuts functional
