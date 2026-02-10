# Execution View

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P1 (High - monitors pipeline execution)
> **Dependencies:** 01-design-system, 02-core-layout, 03-stores-state, 04-tauri-sse-integration
> **Estimated Complexity:** High

## Overview

Implement the live execution monitoring view that shows pipeline progress, preview thumbnails, detection statistics, and checkpoint controls. Receives real-time updates via SSE.

## Goals

- [ ] Multi-step progress bar with current step indicator
- [ ] Live preview grid (6 most recent processed images)
- [ ] Stats dashboard (processed, detected, flagged counts)
- [ ] Checkpoint alert banner with approve/reject
- [ ] ETA and elapsed time display
- [ ] Pause/Resume/Cancel controls
- [ ] Agent commentary stream
- [ ] Error log panel

## Technical Design

### Execution View Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EXECUTION HEADER                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ▶ Executing: Face Anonymization         [Pause] [Cancel]         │  │
│  │  Step 2 of 5: Segment Regions                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  PROGRESS BAR                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  [✓ Detect]──[● Segment]──[○ Anonymize]──[○ Validate]──[○ Export] │  │
│  │                                                                   │  │
│  │  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  42%       │  │
│  │  523 / 1,234 files • ETA: 8 min remaining                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  CHECKPOINT BANNER (when triggered)                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ⚠️ CHECKPOINT: Confidence dropped below threshold                 │  │
│  │  Review the last 10 results before continuing.                    │  │
│  │                                              [Review] [Continue]  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  MAIN CONTENT                                                           │
│  ┌────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │  PREVIEW GRID (2×3)        │  │  STATS PANEL                     │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐│  │  ┌──────────────────────────┐   │  │
│  │  │ img1 │ │ img2 │ │ img3 ││  │  │  📊 Processing Stats     │   │  │
│  │  │ ████ │ │ ████ │ │ ████ ││  │  │                          │   │  │
│  │  └──────┘ └──────┘ └──────┘│  │  │  Processed:    523       │   │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐│  │  │  Detected:   1,847       │   │  │
│  │  │ img4 │ │ img5 │ │ img6 ││  │  │  Flagged:       12       │   │  │
│  │  │ ████ │ │ ████ │ │ ████ ││  │  │  Errors:         3       │   │  │
│  │  └──────┘ └──────┘ └──────┘│  │  │                          │   │  │
│  │                            │  │  │  Avg conf:     0.87      │   │  │
│  │  Click to expand           │  │  │  Throughput: 12 img/s    │   │  │
│  └────────────────────────────┘  │  └──────────────────────────┘   │  │
│                                  │                                  │  │
│                                  │  ┌──────────────────────────┐   │  │
│                                  │  │  🤖 Agent Commentary     │   │  │
│                                  │  │                          │   │  │
│                                  │  │  Processing batch 12...  │   │  │
│                                  │  │  Found 34 faces in last  │   │  │
│                                  │  │  10 images.              │   │  │
│                                  │  │  Confidence stable at    │   │  │
│                                  │  │  0.89 average.           │   │  │
│                                  │  └──────────────────────────┘   │  │
│                                  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ERROR LOG (collapsible, if errors exist)                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ⚠ 3 errors occurred                                         [▼] │  │
│  │  • img_0423.jpg: CUDA out of memory (retrying on CPU)             │  │
│  │  • img_0891.jpg: Corrupt file, skipped                            │  │
│  │  • img_1002.jpg: No faces detected (flagged for review)           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Execution States

```typescript
type ExecutionPhase =
  | 'initializing'   // Loading models, preparing
  | 'running'        // Actively processing
  | 'paused'         // User paused
  | 'checkpoint'     // Awaiting user approval
  | 'completing'     // Finalizing outputs
  | 'completed'      // Done successfully
  | 'cancelled'      // User cancelled
  | 'error';         // Fatal error occurred

interface ExecutionStats {
  processed: number;
  total: number;
  detected: number;
  flagged: number;
  errors: number;
  avg_confidence: number;
  throughput: number;  // items/second
  elapsed_ms: number;
  eta_ms: number;
}
```

### Component Hierarchy

```
ExecutionView.svelte
├── ExecutionHeader.svelte
│   ├── StatusIndicator
│   ├── StepTitle
│   └── ControlButtons (Pause, Cancel)
├── ProgressSection.svelte
│   ├── StepProgress.svelte
│   │   └── StepIndicator.svelte (× n)
│   ├── ProgressBar.svelte
│   └── TimeDisplay.svelte
├── CheckpointBanner.svelte
├── ContentArea.svelte
│   ├── PreviewGrid.svelte
│   │   └── PreviewThumbnail.svelte (× 6)
│   └── StatsPanel.svelte
│       ├── StatCard.svelte (× n)
│       └── CommentaryStream.svelte
└── ErrorLog.svelte
    └── ErrorItem.svelte (× n)
```

## Implementation Tasks

- [ ] **ExecutionView Container**
  - [ ] Create `ExecutionView.svelte`
  - [ ] Connect to `executionStore`
  - [ ] Subscribe to SSE progress events
  - [ ] Handle phase transitions
  - [ ] Auto-navigate on completion

- [ ] **ExecutionHeader Component**
  - [ ] Create `ExecutionHeader.svelte`
  - [ ] Show current phase with icon
  - [ ] Display current step name
  - [ ] Pause button (toggle to Resume)
  - [ ] Cancel button with confirmation

- [ ] **StepProgress Component**
  - [ ] Create `StepProgress.svelte`
  - [ ] Horizontal step indicator
  - [ ] Completed steps: checkmark, filled
  - [ ] Current step: pulsing dot
  - [ ] Future steps: hollow circle
  - [ ] Connecting lines between steps

- [ ] **ProgressBar Component**
  - [ ] Create `ProgressBar.svelte`
  - [ ] Animated fill bar
  - [ ] Percentage display
  - [ ] File count (current/total)
  - [ ] Smooth transitions on updates

- [ ] **TimeDisplay Component**
  - [ ] Create `TimeDisplay.svelte`
  - [ ] Elapsed time counter
  - [ ] ETA calculation and display
  - [ ] Update ETA based on throughput
  - [ ] Show "Calculating..." initially

- [ ] **CheckpointBanner Component**
  - [ ] Create `CheckpointBanner.svelte`
  - [ ] Slide-down animation on trigger
  - [ ] Show checkpoint reason
  - [ ] "Review" button → Review Queue
  - [ ] "Continue" button → resume execution
  - [ ] Auto-dismiss timer (optional)

- [ ] **PreviewGrid Component**
  - [ ] Create `PreviewGrid.svelte`
  - [ ] 2×3 grid of thumbnails
  - [ ] Slide-in animation for new images
  - [ ] Click to expand full-size
  - [ ] Show detection overlays on thumbnails

- [ ] **PreviewThumbnail Component**
  - [ ] Create `PreviewThumbnail.svelte`
  - [ ] Load image from base64 or URL
  - [ ] Draw bounding boxes overlay
  - [ ] Show confidence badge
  - [ ] Loading skeleton state

- [ ] **StatsPanel Component**
  - [ ] Create `StatsPanel.svelte`
  - [ ] Grid of stat cards
  - [ ] Animated number transitions
  - [ ] Color coding (green/yellow/red)
  - [ ] Sparkline for throughput (optional)

- [ ] **CommentaryStream Component**
  - [ ] Create `CommentaryStream.svelte`
  - [ ] Auto-scrolling message list
  - [ ] Timestamp for each message
  - [ ] Truncate old messages

- [ ] **ErrorLog Component**
  - [ ] Create `ErrorLog.svelte`
  - [ ] Collapsible panel
  - [ ] Error count badge
  - [ ] Error severity icons
  - [ ] Copy error details button

## Acceptance Criteria

- [ ] Progress bar updates smoothly via SSE
- [ ] Preview grid shows 6 most recent results
- [ ] Stats update in real-time
- [ ] Checkpoint banner appears and blocks until resolved
- [ ] Pause/Resume works correctly
- [ ] Cancel prompts confirmation and cleans up
- [ ] Errors are displayed without blocking progress
- [ ] View transitions to Review Queue when checkpoint requires review

## Files to Create/Modify

```
src/lib/components/Execution/
├── ExecutionView.svelte
├── ExecutionHeader.svelte
├── ProgressSection.svelte
├── StepProgress.svelte
├── StepIndicator.svelte
├── ProgressBar.svelte
├── TimeDisplay.svelte
├── CheckpointBanner.svelte
├── PreviewGrid.svelte
├── PreviewThumbnail.svelte
├── StatsPanel.svelte
├── StatCard.svelte
├── CommentaryStream.svelte
├── ErrorLog.svelte
├── ErrorItem.svelte
└── index.ts
```

## SSE Event Handling

```typescript
// In ExecutionView.svelte
import { executionStore } from '$lib/stores/execution';
import { sseStore } from '$lib/stores/sse';

$effect(() => {
  // Progress updates
  sseStore.on('progress', (event: ProgressEvent) => {
    executionStore.updateProgress({
      current_step: event.step,
      progress: event.progress,
      processed: event.processed,
      total: event.total,
    });
  });

  // Preview updates
  sseStore.on('preview', (event: PreviewEvent) => {
    executionStore.addPreview({
      file: event.file,
      thumbnail: event.thumbnail,
      detections: event.detections,
    });
  });

  // Checkpoint triggers
  sseStore.on('checkpoint', (event: CheckpointEvent) => {
    executionStore.triggerCheckpoint(event);
    showCheckpointBanner.set(true);
  });

  // Completion
  sseStore.on('complete', () => {
    executionStore.setPhase('completed');
    // Navigate to results or review
  });
});
```

## Progress Bar Animation

```svelte
<!-- ProgressBar.svelte -->
<script lang="ts">
  const { progress, processed, total } = $props<{
    progress: number;
    processed: number;
    total: number;
  }>();

  // Smooth animation using CSS transitions
  let displayProgress = $state(0);

  $effect(() => {
    // Animate to new progress value
    displayProgress = progress;
  });
</script>

<div class="progress-container">
  <div
    class="progress-bar bg-primary transition-all duration-300 ease-out"
    style="width: {displayProgress}%"
  />
  <span class="progress-text">
    {processed.toLocaleString()} / {total.toLocaleString()} files
  </span>
  <span class="progress-percent">{Math.round(displayProgress)}%</span>
</div>

<style>
  .progress-container {
    position: relative;
    height: 24px;
    background: var(--secondary);
    border-radius: var(--radius);
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    min-width: 2%;
  }

  .progress-text {
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    font-size: var(--text-sm);
    color: var(--foreground);
  }

  .progress-percent {
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    font-size: var(--text-sm);
    font-weight: 600;
  }
</style>
```

## Checkpoint Handling Flow

```
1. SSE receives 'checkpoint' event
2. ExecutionStore.triggerCheckpoint() called
3. CheckpointBanner appears with slide animation
4. User clicks "Review" or "Continue"
   - Review: Navigate to Review Queue, execution stays paused
   - Continue: Call resumeProcessing(), hide banner
5. On return from Review Queue, check if still at checkpoint
6. Resume execution with any corrections applied
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause/Resume |
| `Escape` | Cancel (with confirm) |
| `R` | Open Review Queue |
| `Enter` | Continue past checkpoint |
| `E` | Toggle error log |

## Notes

- Throttle preview updates to max 2/second to prevent flooding
- Use `requestAnimationFrame` for smooth progress animations
- Cache thumbnails to prevent flickering on updates
- Consider WebWorker for stats calculations if dataset is large
- Show toast notification when execution completes
