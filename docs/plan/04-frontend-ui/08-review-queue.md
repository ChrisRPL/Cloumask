# Review Queue

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P1 (High - enables quality control)
> **Dependencies:** 01-design-system, 02-core-layout, 03-stores-state
> **Estimated Complexity:** Very High

## Overview

Implement the review queue interface for human-in-the-loop annotation correction. Includes a filterable item list, full-screen annotation canvas with bounding box editing, and keyboard-driven approve/reject/edit workflow.

## Goals

- [ ] Filterable review item list
- [ ] Full-screen annotation canvas
- [ ] Bounding box viewing and editing
- [ ] Polygon/mask viewing (read-only initially)
- [ ] Accept/Reject/Edit keyboard controls
- [ ] Batch operations (approve all, reject all)
- [ ] Undo/redo for edits
- [ ] Progress tracking through queue

## Technical Design

### Review Queue Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  REVIEW HEADER                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ✓ Review Queue              12 / 47 items    [Approve All] [Done]│  │
│  │  Filter: [All ▼] [Flagged ▼] [Confidence < 0.8 ▼]                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  SPLIT VIEW                                                             │
│  ┌────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │  ITEM LIST             │  │  ANNOTATION CANVAS                   │  │
│  │  ┌──────────────────┐  │  │                                      │  │
│  │  │ ▶ img_0423.jpg   │  │  │  ┌────────────────────────────────┐ │  │
│  │  │   ⚠ Low conf     │  │  │  │                                │ │  │
│  │  └──────────────────┘  │  │  │     ┌─────────────┐            │ │  │
│  │  ┌──────────────────┐  │  │  │     │  FACE 0.72  │            │ │  │
│  │  │   img_0424.jpg   │  │  │  │     │   ○───────○ │            │ │  │
│  │  │   ✓ Approved     │  │  │  │     │   │       │ │            │ │  │
│  │  └──────────────────┘  │  │  │     │   ○───────○ │            │ │  │
│  │  ┌──────────────────┐  │  │  │     └─────────────┘            │ │  │
│  │  │   img_0425.jpg   │  │  │  │                                │ │  │
│  │  │   ○ Pending      │  │  │  │          ┌──────────┐         │ │  │
│  │  └──────────────────┘  │  │  │          │ FACE 0.91│         │ │  │
│  │  ┌──────────────────┐  │  │  │          └──────────┘         │ │  │
│  │  │   img_0426.jpg   │  │  │  │                                │ │  │
│  │  │   ○ Pending      │  │  │  └────────────────────────────────┘ │  │
│  │  └──────────────────┘  │  │                                      │  │
│  │                        │  │  [Zoom: 100%] [Fit] [1:1]            │  │
│  │  [Load more...]        │  │                                      │  │
│  └────────────────────────┘  └──────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ANNOTATION DETAILS                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  img_0423.jpg • 1920×1080 • 2 annotations                         │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────┐    │  │
│  │  │ ▶ Face #1               │  │   Face #2                   │    │  │
│  │  │   Confidence: 0.72      │  │   Confidence: 0.91          │    │  │
│  │  │   Box: 120,80,200,180   │  │   Box: 450,100,150,160      │    │  │
│  │  │   [Edit] [Delete]       │  │   [Edit] [Delete]           │    │  │
│  │  └─────────────────────────┘  └─────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ACTION BAR                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  [← Prev]  [A] Accept  [R] Reject  [E] Edit  [Next →]   12/47    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Review Item States

```typescript
type ReviewStatus =
  | 'pending'     // Not yet reviewed
  | 'approved'    // Accepted as-is
  | 'rejected'    // Marked for exclusion
  | 'edited';     // Annotations modified

interface ReviewItem {
  id: string;
  file_path: string;
  file_name: string;
  dimensions: { width: number; height: number };
  thumbnail_url: string;
  full_image_url: string;
  annotations: Annotation[];
  original_annotations: Annotation[];  // For undo
  confidence: number;  // Min confidence of annotations
  flagged: boolean;
  flag_reason?: string;
  status: ReviewStatus;
  reviewed_at?: Date;
}

interface Annotation {
  id: string;
  type: 'bbox' | 'polygon' | 'mask';
  class_name: string;
  confidence: number;
  bbox?: { x: number; y: number; width: number; height: number };
  polygon?: { x: number; y: number }[];
  mask_url?: string;
  color: string;
  visible: boolean;
  edited: boolean;
}
```

### Component Hierarchy

```
ReviewQueue.svelte
├── ReviewHeader.svelte
│   ├── Title & Progress
│   ├── FilterBar.svelte
│   │   └── FilterDropdown.svelte (× n)
│   └── BatchActions
├── SplitPane.svelte
│   ├── ItemList.svelte
│   │   └── ReviewListItem.svelte (× n)
│   └── AnnotationCanvas.svelte
│       ├── ImageLayer.svelte
│       ├── AnnotationLayer.svelte
│       │   └── BoundingBox.svelte (× n)
│       ├── CanvasControls.svelte
│       └── ZoomControls.svelte
├── AnnotationDetails.svelte
│   └── AnnotationCard.svelte (× n)
└── ActionBar.svelte
    └── ActionButton.svelte (× n)
```

## Implementation Tasks

- [ ] **ReviewQueue Container**
  - [ ] Create `ReviewQueue.svelte`
  - [ ] Connect to `reviewStore`
  - [ ] Handle item selection
  - [ ] Manage keyboard shortcuts
  - [ ] Track review progress

- [ ] **ReviewHeader Component**
  - [ ] Create `ReviewHeader.svelte`
  - [ ] Progress display (X / Y items)
  - [ ] "Approve All" batch action
  - [ ] "Done" button to finish review
  - [ ] Return to Execution View option

- [ ] **FilterBar Component**
  - [ ] Create `FilterBar.svelte`
  - [ ] Status filter (All, Pending, Approved, Rejected, Edited)
  - [ ] Flag filter (All, Flagged, Not flagged)
  - [ ] Confidence filter (threshold slider)
  - [ ] Class filter (detected classes)
  - [ ] Search by filename

- [ ] **ItemList Component**
  - [ ] Create `ItemList.svelte`
  - [ ] Virtual scrolling for large lists
  - [ ] Thumbnail + status indicator
  - [ ] Selection highlight
  - [ ] Keyboard navigation (↑/↓)
  - [ ] "Load more" pagination

- [ ] **AnnotationCanvas Component**
  - [ ] Create `AnnotationCanvas.svelte`
  - [ ] Load and display image
  - [ ] Zoom and pan controls
  - [ ] Fit to container / 1:1 modes
  - [ ] Mouse wheel zoom
  - [ ] Drag to pan

- [ ] **AnnotationLayer Component**
  - [ ] Create `AnnotationLayer.svelte`
  - [ ] SVG overlay for annotations
  - [ ] Render bounding boxes
  - [ ] Render polygons (view-only)
  - [ ] Show class labels and confidence
  - [ ] Hover highlight effect

- [ ] **BoundingBox Component**
  - [ ] Create `BoundingBox.svelte`
  - [ ] Rectangular box rendering
  - [ ] Resize handles (8 points)
  - [ ] Drag to move
  - [ ] Delete button
  - [ ] Class label display
  - [ ] Selection state

- [ ] **BoundingBox Editing**
  - [ ] Click to select box
  - [ ] Drag corners/edges to resize
  - [ ] Drag center to move
  - [ ] Minimum size constraint
  - [ ] Snap to edges (optional)
  - [ ] Real-time coordinate display

- [ ] **AnnotationDetails Component**
  - [ ] Create `AnnotationDetails.svelte`
  - [ ] List all annotations for current item
  - [ ] Editable class name dropdown
  - [ ] Coordinate display
  - [ ] Delete annotation button
  - [ ] Add new annotation button

- [ ] **ActionBar Component**
  - [ ] Create `ActionBar.svelte`
  - [ ] Previous/Next navigation
  - [ ] Accept (A), Reject (R), Edit (E) buttons
  - [ ] Current position indicator
  - [ ] Keyboard hint display

- [ ] **Undo/Redo System**
  - [ ] Implement edit history stack
  - [ ] Undo last annotation change
  - [ ] Redo after undo
  - [ ] Reset to original annotations
  - [ ] Keyboard shortcuts (Ctrl+Z, Ctrl+Y)

- [ ] **Batch Operations**
  - [ ] Approve all pending items
  - [ ] Reject all flagged items
  - [ ] Confirmation dialog
  - [ ] Progress indicator

## Acceptance Criteria

- [ ] Review queue shows filtered list of items
- [ ] Selecting item shows annotations on canvas
- [ ] Bounding boxes can be resized and moved
- [ ] A/R/E keyboard shortcuts work
- [ ] Progress persists across sessions
- [ ] Undo/redo works for annotation edits
- [ ] Batch approve/reject works
- [ ] Can navigate with keyboard only

## Files to Create/Modify

```
src/lib/components/Review/
├── ReviewQueue.svelte
├── ReviewHeader.svelte
├── FilterBar.svelte
├── FilterDropdown.svelte
├── ItemList.svelte
├── ReviewListItem.svelte
├── SplitPane.svelte
├── AnnotationCanvas.svelte
├── ImageLayer.svelte
├── AnnotationLayer.svelte
├── BoundingBox.svelte
├── CanvasControls.svelte
├── ZoomControls.svelte
├── AnnotationDetails.svelte
├── AnnotationCard.svelte
├── ActionBar.svelte
├── ActionButton.svelte
└── index.ts

src/lib/stores/
└── review.ts              # Review state management

src/lib/utils/
└── canvas.ts              # Canvas utilities (zoom, pan, etc.)
```

## Bounding Box Editing Implementation

```svelte
<!-- BoundingBox.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{
    update: { id: string; bbox: BBox };
    delete: { id: string };
    select: { id: string };
  }>();

  const { annotation, scale, selected } = $props<{
    annotation: Annotation;
    scale: number;
    selected: boolean;
  }>();

  let isDragging = $state(false);
  let dragType = $state<'move' | 'resize' | null>(null);
  let dragHandle = $state<string | null>(null);

  const handles = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

  function handleMouseDown(e: MouseEvent, type: 'move' | 'resize', handle?: string) {
    e.stopPropagation();
    isDragging = true;
    dragType = type;
    dragHandle = handle ?? null;
    dispatch('select', { id: annotation.id });
  }

  function handleMouseMove(e: MouseEvent) {
    if (!isDragging || !annotation.bbox) return;

    const dx = e.movementX / scale;
    const dy = e.movementY / scale;

    if (dragType === 'move') {
      annotation.bbox.x += dx;
      annotation.bbox.y += dy;
    } else if (dragType === 'resize' && dragHandle) {
      // Handle resize based on which handle is being dragged
      resizeBox(dragHandle, dx, dy);
    }

    dispatch('update', { id: annotation.id, bbox: annotation.bbox });
  }

  function resizeBox(handle: string, dx: number, dy: number) {
    const box = annotation.bbox!;

    switch (handle) {
      case 'nw':
        box.x += dx; box.y += dy;
        box.width -= dx; box.height -= dy;
        break;
      case 'ne':
        box.y += dy;
        box.width += dx; box.height -= dy;
        break;
      case 'se':
        box.width += dx; box.height += dy;
        break;
      case 'sw':
        box.x += dx;
        box.width -= dx; box.height += dy;
        break;
      // ... other handles
    }

    // Enforce minimum size
    box.width = Math.max(20, box.width);
    box.height = Math.max(20, box.height);
  }
</script>

<g
  class="bounding-box"
  class:selected
  on:mousedown={(e) => handleMouseDown(e, 'move')}
>
  <!-- Main rectangle -->
  <rect
    x={annotation.bbox.x * scale}
    y={annotation.bbox.y * scale}
    width={annotation.bbox.width * scale}
    height={annotation.bbox.height * scale}
    fill="none"
    stroke={annotation.color}
    stroke-width={selected ? 3 : 2}
  />

  <!-- Label -->
  <text
    x={annotation.bbox.x * scale}
    y={annotation.bbox.y * scale - 5}
    class="label"
  >
    {annotation.class_name} {(annotation.confidence * 100).toFixed(0)}%
  </text>

  <!-- Resize handles (only when selected) -->
  {#if selected}
    {#each handles as handle}
      <circle
        class="handle"
        cx={getHandleX(handle)}
        cy={getHandleY(handle)}
        r={6}
        on:mousedown={(e) => handleMouseDown(e, 'resize', handle)}
      />
    {/each}
  {/if}
</g>
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` | Approve current item |
| `R` | Reject current item |
| `E` | Enter edit mode |
| `↑` / `K` | Previous item |
| `↓` / `J` | Next item |
| `Delete` | Delete selected annotation |
| `N` | Add new bounding box |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+Shift+Z` | Redo (alt) |
| `Escape` | Exit edit mode / Deselect |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `0` | Fit to view |
| `1` | 100% zoom |

## Notes

- Consider using Fabric.js or Konva for canvas interactions
- Image should be loaded lazily to avoid memory issues
- Annotations should be saved on item change (debounced)
- Support touch gestures for tablet users
- Add "flag for expert review" option
- Consider side-by-side before/after comparison view
