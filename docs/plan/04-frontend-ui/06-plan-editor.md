# Plan Editor

> **Status:** 🔴 Not Started
> **Priority:** P1 (High - enables pipeline configuration)
> **Dependencies:** 01-design-system, 02-core-layout, 03-stores-state
> **Estimated Complexity:** High

## Overview

Implement the visual pipeline editor where users can view, modify, and approve processing plans created by the agent. Supports drag-and-drop reordering, step configuration, and execution controls.

## Goals

- [ ] Visual pipeline step list
- [ ] Drag-and-drop step reordering
- [ ] Step configuration panels
- [ ] Enable/disable individual steps
- [ ] Approve/Edit/Cancel controls
- [ ] Estimated time and resource display
- [ ] Step dependency visualization

## Technical Design

### Plan Editor Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PLAN HEADER                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  📋 Processing Plan                      [Edit] [Cancel] [Start]  │  │
│  │  Face Anonymization Pipeline • 5 steps • ~12 min                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  PIPELINE VISUALIZATION                                                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐      │  │
│  │  │ 1. Detect    │────▶│ 2. Segment   │────▶│ 3. Anonymize │      │  │
│  │  │    Faces     │     │    Regions   │     │    Blur      │      │  │
│  │  │  YOLO11m     │     │    SAM2      │     │  Gaussian    │      │  │
│  │  └──────────────┘     └──────────────┘     └──────────────┘      │  │
│  │          │                                         │              │  │
│  │          ▼                                         ▼              │  │
│  │  ┌──────────────┐                         ┌──────────────┐       │  │
│  │  │ 4. Validate  │                         │ 5. Export    │       │  │
│  │  │    Results   │────────────────────────▶│    COCO      │       │  │
│  │  │  Confidence  │                         │    Format    │       │  │
│  │  └──────────────┘                         └──────────────┘       │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  STEP LIST (drag to reorder)                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  [☰] [✓] 1. Detect Faces           YOLO11m • conf: 0.5      [⚙]  │  │
│  │  [☰] [✓] 2. Segment Regions        SAM2 • auto              [⚙]  │  │
│  │  [☰] [✓] 3. Anonymize              Gaussian blur • r: 21    [⚙]  │  │
│  │  [☰] [✓] 4. Validate Results       conf > 0.8               [⚙]  │  │
│  │  [☰] [✓] 5. Export                 COCO JSON                [⚙]  │  │
│  │                                                                   │  │
│  │  [+ Add Step]                                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  STEP CONFIGURATION (when step selected)                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ⚙ Configure: Detect Faces                                        │  │
│  │                                                                   │  │
│  │  Model:      [YOLO11m          ▼]                                │  │
│  │  Confidence: [====●============] 0.50                            │  │
│  │  Classes:    [☑ Face] [☑ Person] [☐ Car] [☐ License]             │  │
│  │  GPU:        [☑ Use if available]                                │  │
│  │                                                                   │  │
│  │  [Reset to defaults]                            [Apply changes]  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step Types and Configurations

```typescript
// Step type definitions with their config schemas
const stepSchemas: Record<StepType, StepSchema> = {
  detect: {
    name: 'Detect Objects',
    icon: 'Search',
    config: {
      model: { type: 'select', options: ['YOLO11m', 'YOLO-World', 'RT-DETR', 'GroundingDINO'] },
      confidence: { type: 'slider', min: 0.1, max: 1.0, step: 0.05, default: 0.5 },
      classes: { type: 'multi-select', options: ['face', 'person', 'car', 'license_plate'] },
      use_gpu: { type: 'checkbox', default: true },
    },
  },
  segment: {
    name: 'Segment Regions',
    icon: 'Scissors',
    config: {
      model: { type: 'select', options: ['SAM2', 'SAM3', 'MobileSAM'] },
      mode: { type: 'select', options: ['auto', 'point', 'box'] },
      mask_threshold: { type: 'slider', min: 0.0, max: 1.0, step: 0.05, default: 0.5 },
    },
  },
  anonymize: {
    name: 'Anonymize',
    icon: 'EyeOff',
    config: {
      method: { type: 'select', options: ['gaussian_blur', 'pixelate', 'solid_fill', 'inpaint'] },
      blur_radius: { type: 'slider', min: 5, max: 51, step: 2, default: 21 },
      fill_color: { type: 'color', default: '#000000' },
    },
  },
  label: {
    name: 'Auto-Label',
    icon: 'Tag',
    config: {
      format: { type: 'select', options: ['COCO', 'YOLO', 'Pascal VOC', 'LabelMe'] },
      include_confidence: { type: 'checkbox', default: true },
      min_confidence: { type: 'slider', min: 0.0, max: 1.0, step: 0.05, default: 0.0 },
    },
  },
  export: {
    name: 'Export',
    icon: 'Download',
    config: {
      format: { type: 'select', options: ['COCO', 'YOLO', 'Pascal VOC', 'CSV', 'JSON'] },
      output_dir: { type: 'path', default: './output' },
      include_images: { type: 'checkbox', default: true },
      zip: { type: 'checkbox', default: false },
    },
  },
  validate: {
    name: 'Validate',
    icon: 'CheckCircle',
    config: {
      min_confidence: { type: 'slider', min: 0.0, max: 1.0, step: 0.05, default: 0.8 },
      flag_uncertain: { type: 'checkbox', default: true },
      auto_skip_low: { type: 'checkbox', default: false },
    },
  },
  transform: {
    name: 'Transform',
    icon: 'Wand',
    config: {
      resize: { type: 'dimensions', default: { width: null, height: null } },
      rotate: { type: 'select', options: ['none', '90', '180', '270', 'auto'] },
      format: { type: 'select', options: ['keep', 'jpg', 'png', 'webp'] },
    },
  },
};
```

### Component Hierarchy

```
PlanEditor.svelte
├── PlanHeader.svelte
│   ├── PlanTitle
│   ├── PlanStats (steps, time estimate)
│   └── ActionButtons (Edit, Cancel, Start)
├── PipelineVisualizer.svelte
│   └── StepNode.svelte (× n)
│       └── StepConnector.svelte
├── StepList.svelte
│   └── StepListItem.svelte (× n)
│       ├── DragHandle
│       ├── EnableToggle
│       ├── StepSummary
│       └── ConfigButton
├── AddStepButton.svelte
└── StepConfig.svelte
    ├── ConfigField.svelte (× n)
    └── ConfigActions
```

## Implementation Tasks

- [ ] **PlanEditor Container**
  - [ ] Create `PlanEditor.svelte` main component
  - [ ] Connect to `pipelineStore`
  - [ ] Handle plan approval workflow
  - [ ] Manage step selection state

- [ ] **PlanHeader Component**
  - [ ] Create `PlanHeader.svelte`
  - [ ] Show plan name/description
  - [ ] Display step count and time estimate
  - [ ] Edit mode toggle
  - [ ] Cancel plan button
  - [ ] Start execution button

- [ ] **PipelineVisualizer Component**
  - [ ] Create `PipelineVisualizer.svelte`
  - [ ] Render steps as connected nodes
  - [ ] Show step types with icons
  - [ ] Highlight selected step
  - [ ] Animate flow direction
  - [ ] Handle branching pipelines

- [ ] **StepList Component**
  - [ ] Create `StepList.svelte`
  - [ ] Implement drag-and-drop with `svelte-dnd-action`
  - [ ] Enable/disable step checkboxes
  - [ ] Step summary with key config values
  - [ ] Config button to open panel

- [ ] **StepListItem Component**
  - [ ] Create `StepListItem.svelte`
  - [ ] Drag handle (6-dot grip icon)
  - [ ] Step number indicator
  - [ ] Step type icon and name
  - [ ] Model/config summary
  - [ ] Delete step button (in edit mode)

- [ ] **StepConfig Component**
  - [ ] Create `StepConfig.svelte`
  - [ ] Dynamic form based on step type
  - [ ] Select dropdowns for models
  - [ ] Sliders for numeric values
  - [ ] Checkboxes for boolean options
  - [ ] File picker for paths
  - [ ] Reset and Apply buttons

- [ ] **ConfigField Component**
  - [ ] Create `ConfigField.svelte`
  - [ ] Type-specific input rendering
  - [ ] Validation display
  - [ ] Default value indication
  - [ ] Help tooltip with description

- [ ] **AddStepButton Component**
  - [ ] Create `AddStepButton.svelte`
  - [ ] Dropdown with available step types
  - [ ] Insert at end or at position
  - [ ] Keyboard shortcut support

- [ ] **Drag-and-Drop**
  - [ ] Install `svelte-dnd-action`
  - [ ] Implement step reordering
  - [ ] Visual feedback during drag
  - [ ] Update store on drop
  - [ ] Validate step dependencies

## Acceptance Criteria

- [ ] Pipeline steps display in visual and list form
- [ ] Steps can be reordered via drag-and-drop
- [ ] Step configuration can be modified
- [ ] Individual steps can be enabled/disabled
- [ ] "Start" transitions to Execution View
- [ ] "Cancel" returns to Chat with message
- [ ] Time estimates update when config changes

## Files to Create/Modify

```
src/lib/components/Plan/
├── PlanEditor.svelte
├── PlanHeader.svelte
├── PipelineVisualizer.svelte
├── StepNode.svelte
├── StepConnector.svelte
├── StepList.svelte
├── StepListItem.svelte
├── StepConfig.svelte
├── ConfigField.svelte
├── AddStepButton.svelte
└── index.ts

src/lib/utils/
└── step-schemas.ts        # Step type configurations
```

## Drag-and-Drop Implementation

```svelte
<!-- StepList.svelte -->
<script lang="ts">
  import { dndzone } from 'svelte-dnd-action';
  import { pipelineStore } from '$lib/stores/pipeline';
  import StepListItem from './StepListItem.svelte';

  let steps = $derived($pipelineStore.steps);

  function handleDndConsider(e: CustomEvent) {
    steps = e.detail.items;
  }

  function handleDndFinalize(e: CustomEvent) {
    steps = e.detail.items;
    pipelineStore.reorderSteps(steps.map(s => s.id));
  }
</script>

<div
  class="step-list"
  use:dndzone={{ items: steps, flipDurationMs: 200 }}
  on:consider={handleDndConsider}
  on:finalize={handleDndFinalize}
>
  {#each steps as step (step.id)}
    <StepListItem {step} />
  {/each}
</div>
```

## Time Estimation

```typescript
// src/lib/utils/time-estimate.ts

interface TimeFactors {
  base_ms: number;         // Base time per item
  model_multiplier: number; // Multiplier based on model complexity
  gpu_speedup: number;     // GPU vs CPU speedup factor
}

const stepTimeFactors: Record<StepType, TimeFactors> = {
  detect: { base_ms: 50, model_multiplier: 1.0, gpu_speedup: 10 },
  segment: { base_ms: 200, model_multiplier: 1.5, gpu_speedup: 15 },
  anonymize: { base_ms: 20, model_multiplier: 1.0, gpu_speedup: 5 },
  label: { base_ms: 5, model_multiplier: 1.0, gpu_speedup: 1 },
  export: { base_ms: 10, model_multiplier: 1.0, gpu_speedup: 1 },
  validate: { base_ms: 2, model_multiplier: 1.0, gpu_speedup: 1 },
  transform: { base_ms: 30, model_multiplier: 1.0, gpu_speedup: 3 },
};

export function estimatePipelineTime(
  steps: PipelineStep[],
  itemCount: number,
  hasGPU: boolean
): number {
  return steps
    .filter(s => s.enabled)
    .reduce((total, step) => {
      const factors = stepTimeFactors[step.type];
      const stepTime = factors.base_ms * factors.model_multiplier;
      const adjustedTime = hasGPU ? stepTime / factors.gpu_speedup : stepTime;
      return total + (adjustedTime * itemCount);
    }, 0);
}
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑/↓` | Navigate steps |
| `Space` | Toggle step enabled |
| `Enter` | Open step config |
| `Delete` | Remove step (edit mode) |
| `Ctrl+↑/↓` | Move step up/down |
| `Escape` | Close config panel |
| `Ctrl+Enter` | Start execution |

## Notes

- Use CSS transitions for smooth reorder animations
- Persist draft plans to localStorage for recovery
- Show warning if removing a step breaks dependencies
- Consider adding "duplicate step" functionality
- Export plan as JSON for sharing/backup
