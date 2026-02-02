# Frontend UI Module

> **Status:** 🟢 Complete
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation, 02-agent-system

## Overview

Build the Svelte 5 frontend with five main views: Chat, Plan Editor, Execution View, Review Queue, and Point Cloud Viewer. Dark mode by default, keyboard-first navigation.

## Goals

- [x] Conversational chat interface with streaming responses
- [x] Visual pipeline plan editor
- [x] Live execution view with preview grid
- [x] Review queue for annotation corrections
- [x] Three.js point cloud viewer
- [x] Responsive dark theme UI

## Technical Design

### Design System
- **Framework:** Svelte 5 with runes ($state, $derived, $effect)
- **Components:** shadcn/ui (bits-ui + Tailwind)
- **Icons:** Lucide
- **Theme:** Dark mode default, violet accent (#8b5cf6)

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

- [ ] **Core Layout**
  - [ ] App shell with sidebar navigation
  - [ ] Dark theme configuration
  - [ ] Responsive breakpoints
  - [ ] Keyboard navigation (vim-like)

- [ ] **Chat Panel**
  - [ ] Message list with user/agent distinction
  - [ ] Streaming text display
  - [ ] Inline clarification questions
  - [ ] Quick-reply buttons
  - [ ] Plan preview in messages

- [ ] **Plan Editor**
  - [ ] Pipeline step visualization
  - [ ] Drag-and-drop reordering
  - [ ] Step configuration panels
  - [ ] Start/Edit/Cancel CTAs

- [ ] **Execution View**
  - [ ] Progress bar with step indicators
  - [ ] Live preview grid (6 images)
  - [ ] Checkpoint alert banner
  - [ ] Stats dashboard (processed, detected, flagged)
  - [ ] Agent commentary stream

- [ ] **Review Queue**
  - [ ] Filterable item list
  - [ ] Full-screen annotation canvas
  - [ ] Bounding box editing
  - [ ] Accept/Reject/Edit controls
  - [ ] Keyboard shortcuts (A/R/E)

- [ ] **Stores & State**
  - [ ] `agent.ts` - Agent state, messages
  - [ ] `pipeline.ts` - Current pipeline state
  - [ ] `settings.ts` - User preferences
  - [ ] `sse.ts` - SSE connection management

## Acceptance Criteria

- [ ] Chat shows streaming responses from agent
- [ ] Plan editor displays pipeline steps visually
- [ ] Execution view updates in real-time during processing
- [ ] Review queue allows editing bounding boxes
- [ ] All views navigable via keyboard
- [ ] Dark mode renders correctly on all views

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

- [ ] 01-design-system - Tailwind + shadcn/ui configured
- [ ] 02-core-layout - App shell functional
- [ ] 03-stores-state - All stores implemented
- [ ] 04-tauri-sse-integration - Backend communication working
- [ ] 05-chat-panel - Chat with streaming working
- [ ] 06-plan-editor - Plan editing functional
- [ ] 07-execution-view - Live progress updates
- [ ] 08-review-queue - Annotation editing works
- [ ] 09-pointcloud-viewer - 3D viewer renders
- [ ] 10-keyboard-navigation - All shortcuts functional
