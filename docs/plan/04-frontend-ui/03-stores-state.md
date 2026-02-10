# Stores & State Management

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P0 (Critical - provides data layer for all views)
> **Dependencies:** 01-design-system, 02-core-layout
> **Estimated Complexity:** High

## Overview

Implement Svelte stores for global application state: agent messages, pipeline configuration, execution state, user settings, and UI state. Uses Svelte 5 runes pattern with reactive stores for cross-component communication.

## Goals

- [ ] Agent store for chat messages and conversation state
- [ ] Pipeline store for current plan and step configurations
- [ ] Execution store for processing progress and results
- [ ] Settings store for user preferences
- [ ] UI store for navigation and layout state
- [ ] Persistence layer for settings and drafts

## Technical Design

### Store Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application State                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Agent     │  │  Pipeline   │  │      Execution          │ │
│  │   Store     │  │   Store     │  │       Store             │ │
│  │             │  │             │  │                         │ │
│  │ • messages  │  │ • steps     │  │ • status                │ │
│  │ • state     │  │ • config    │  │ • progress              │ │
│  │ • context   │  │ • selected  │  │ • results               │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Settings   │  │     UI      │  │       Review            │ │
│  │   Store     │  │   Store     │  │       Store             │ │
│  │             │  │             │  │                         │ │
│  │ • theme     │  │ • view      │  │ • items                 │ │
│  │ • shortcuts │  │ • sidebar   │  │ • selected              │ │
│  │ • defaults  │  │ • modals    │  │ • filters               │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Type Definitions

```typescript
// src/lib/types/agent.ts
export interface Message {
  id: string;
  role: 'user' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    tool_calls?: ToolCall[];
    plan_preview?: PipelineStep[];
    clarification?: ClarificationRequest;
  };
}

export interface ClarificationRequest {
  question: string;
  options: string[];
  required: boolean;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  result?: unknown;
  status: 'pending' | 'running' | 'completed' | 'error';
}

export type AgentState =
  | 'idle'
  | 'understanding'
  | 'planning'
  | 'awaiting_approval'
  | 'executing'
  | 'checkpoint'
  | 'complete'
  | 'error';

// src/lib/types/pipeline.ts
export interface PipelineStep {
  id: string;
  type: StepType;
  name: string;
  config: StepConfig;
  order: number;
  enabled: boolean;
  estimated_duration?: number;
}

export type StepType =
  | 'detect'
  | 'segment'
  | 'anonymize'
  | 'label'
  | 'export'
  | 'validate'
  | 'transform';

export interface StepConfig {
  model?: string;
  confidence?: number;
  classes?: string[];
  output_format?: string;
  [key: string]: unknown;
}

// src/lib/types/execution.ts
export interface ExecutionState {
  status: ExecutionStatus;
  current_step: number;
  total_steps: number;
  progress: number;  // 0-100
  processed: number;
  total: number;
  detected: number;
  flagged: number;
  errors: ExecutionError[];
  start_time?: Date;
  eta?: Date;
}

export type ExecutionStatus =
  | 'idle'
  | 'running'
  | 'paused'
  | 'checkpoint'
  | 'completed'
  | 'error';

export interface ExecutionError {
  step: number;
  file: string;
  message: string;
  recoverable: boolean;
}

// src/lib/types/review.ts
export interface ReviewItem {
  id: string;
  file_path: string;
  thumbnail_url: string;
  annotations: Annotation[];
  confidence: number;
  flagged: boolean;
  status: 'pending' | 'approved' | 'rejected' | 'edited';
}

export interface Annotation {
  id: string;
  type: 'bbox' | 'polygon' | 'mask';
  class: string;
  confidence: number;
  coordinates: number[];
  edited?: boolean;
}
```

### Store Implementations

```typescript
// src/lib/stores/agent.ts
import { writable, derived } from 'svelte/store';
import type { Message, AgentState } from '$lib/types/agent';

function createAgentStore() {
  const messages = writable<Message[]>([]);
  const state = writable<AgentState>('idle');
  const conversationId = writable<string | null>(null);

  // Derived: is agent busy?
  const isBusy = derived(state, $state =>
    ['understanding', 'planning', 'executing'].includes($state)
  );

  // Derived: needs user input?
  const needsInput = derived(state, $state =>
    ['awaiting_approval', 'checkpoint'].includes($state)
  );

  return {
    messages,
    state,
    conversationId,
    isBusy,
    needsInput,

    // Actions
    addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => {
      messages.update(msgs => [...msgs, {
        ...message,
        id: crypto.randomUUID(),
        timestamp: new Date(),
      }]);
    },

    clearMessages: () => messages.set([]),

    setState: (newState: AgentState) => state.set(newState),

    startConversation: () => {
      conversationId.set(crypto.randomUUID());
      messages.set([]);
      state.set('idle');
    },
  };
}

export const agentStore = createAgentStore();
```

## Implementation Tasks

- [ ] **Type Definitions**
  - [ ] Create `src/lib/types/agent.ts`
  - [ ] Create `src/lib/types/pipeline.ts`
  - [ ] Create `src/lib/types/execution.ts`
  - [ ] Create `src/lib/types/review.ts`
  - [ ] Create `src/lib/types/settings.ts`
  - [ ] Create `src/lib/types/index.ts` (re-exports)

- [ ] **Agent Store**
  - [ ] Implement message list with add/clear
  - [ ] Implement state machine transitions
  - [ ] Add conversation ID management
  - [ ] Create derived stores (isBusy, needsInput)
  - [ ] Add streaming message support (partial updates)

- [ ] **Pipeline Store**
  - [ ] Implement step list with CRUD operations
  - [ ] Add drag-and-drop reordering support
  - [ ] Implement step configuration updates
  - [ ] Add validation for step dependencies
  - [ ] Create derived store for enabled steps only

- [ ] **Execution Store**
  - [ ] Implement progress tracking
  - [ ] Add stats counters (processed, detected, flagged)
  - [ ] Implement error collection
  - [ ] Add ETA calculation
  - [ ] Create checkpoint state management

- [ ] **Review Store**
  - [ ] Implement review item list
  - [ ] Add filter/sort capabilities
  - [ ] Track selection state
  - [ ] Implement batch operations (approve all, reject all)
  - [ ] Add undo/redo for annotation edits

- [ ] **Settings Store**
  - [ ] Define default settings
  - [ ] Implement localStorage persistence
  - [ ] Add theme preference (dark/light/system)
  - [ ] Add keyboard shortcuts customization
  - [ ] Add model defaults (confidence thresholds)

- [ ] **UI Store**
  - [ ] Current view state
  - [ ] Sidebar expanded state
  - [ ] Modal/dialog stack
  - [ ] Toast notifications queue
  - [ ] Loading states per-view

- [ ] **Persistence Layer**
  - [ ] Create `src/lib/utils/persistence.ts`
  - [ ] Implement localStorage adapter
  - [ ] Add Tauri file system adapter (for project data)
  - [ ] Create auto-save for settings
  - [ ] Add draft message persistence

## Acceptance Criteria

- [ ] All stores export typed interfaces
- [ ] State changes trigger reactive updates
- [ ] Settings persist across app restarts
- [ ] Stores can be subscribed to from any component
- [ ] Derived stores update efficiently
- [ ] No memory leaks on store subscriptions

## Files to Create/Modify

```
src/lib/
├── types/
│   ├── agent.ts
│   ├── pipeline.ts
│   ├── execution.ts
│   ├── review.ts
│   ├── settings.ts
│   └── index.ts
├── stores/
│   ├── agent.ts
│   ├── pipeline.ts
│   ├── execution.ts
│   ├── review.ts
│   ├── settings.ts
│   ├── ui.ts
│   └── index.ts
└── utils/
    └── persistence.ts
```

## Svelte 5 Runes Usage

```svelte
<script lang="ts">
  import { agentStore } from '$lib/stores/agent';

  // Subscribe to store values using $
  const messages = $derived(agentStore.messages);
  const isBusy = $derived(agentStore.isBusy);

  // Local state with $state
  let input = $state('');

  // Effects for side-effects
  $effect(() => {
    if ($isBusy) {
      // Show loading indicator
    }
  });
</script>
```

## Store Subscriptions Pattern

```typescript
// In Svelte 5 components, use $derived for store values
// In plain TypeScript, use .subscribe()

import { agentStore } from '$lib/stores/agent';

// TypeScript (outside components)
const unsubscribe = agentStore.messages.subscribe(msgs => {
  console.log('Messages updated:', msgs.length);
});

// Cleanup
unsubscribe();
```

## Notes

- Use `crypto.randomUUID()` for generating IDs (browser-native)
- Consider immer for immutable updates on complex nested state
- Settings store should debounce localStorage writes
- Review store may need pagination for large datasets
- All timestamps should be ISO 8601 strings for serialization
