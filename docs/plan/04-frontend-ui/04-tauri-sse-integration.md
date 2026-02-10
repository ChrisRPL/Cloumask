# Tauri IPC & SSE Integration

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P0 (Critical - enables backend communication)
> **Dependencies:** 01-design-system, 03-stores-state, 01-foundation (Tauri setup)
> **Estimated Complexity:** High

## Overview

Implement the communication layer between the Svelte frontend and Rust/Python backends. This includes Tauri IPC command wrappers, Server-Sent Events (SSE) for real-time streaming, and error handling patterns.

## Goals

- [ ] Type-safe Tauri IPC command wrappers
- [ ] SSE connection manager for streaming updates
- [ ] Automatic reconnection with backoff
- [ ] Error boundary and retry logic
- [ ] Request/response correlation
- [ ] Connection status indicator

## Technical Design

### Communication Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Svelte Frontend                               │
│  ┌─────────────────────┐  ┌─────────────────────┐                       │
│  │   Tauri IPC Client  │  │   SSE Client        │                       │
│  │   (Request/Reply)   │  │   (Push/Stream)     │                       │
│  └──────────┬──────────┘  └──────────┬──────────┘                       │
└─────────────┼────────────────────────┼──────────────────────────────────┘
              │                        │
              ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tauri Rust Shell                                │
│  ┌─────────────────────┐  ┌─────────────────────┐                       │
│  │   IPC Handlers      │  │   SSE Proxy         │                       │
│  │   (Commands)        │  │   (Forward to FE)   │                       │
│  └──────────┬──────────┘  └──────────┬──────────┘                       │
└─────────────┼────────────────────────┼──────────────────────────────────┘
              │                        │
              ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Python Sidecar (FastAPI)                          │
│  ┌─────────────────────┐  ┌─────────────────────┐                       │
│  │   REST Endpoints    │  │   SSE Endpoints     │                       │
│  │   /api/*            │  │   /stream/*         │                       │
│  └─────────────────────┘  └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tauri Command Types

```typescript
// src/lib/types/ipc.ts

// File operations
export interface FileSelection {
  paths: string[];
  directory?: string;
}

export interface FileInfo {
  path: string;
  name: string;
  size: number;
  modified: Date;
  is_directory: boolean;
}

// Project operations
export interface Project {
  id: string;
  name: string;
  path: string;
  created: Date;
  last_opened: Date;
}

// Processing commands
export interface ProcessRequest {
  files: string[];
  pipeline: PipelineStep[];
  options: ProcessOptions;
}

export interface ProcessOptions {
  output_dir: string;
  overwrite: boolean;
  checkpoint_interval: number;
}

// Health check
export interface HealthStatus {
  sidecar: 'running' | 'stopped' | 'error';
  models_loaded: string[];
  gpu_available: boolean;
  memory_usage: number;
}
```

### SSE Event Types

```typescript
// src/lib/types/sse.ts

export type SSEEventType =
  | 'agent_state'
  | 'agent_message'
  | 'progress'
  | 'preview'
  | 'checkpoint'
  | 'error'
  | 'complete';

export interface SSEEvent<T = unknown> {
  type: SSEEventType;
  data: T;
  timestamp: string;
  correlation_id?: string;
}

// Specific event payloads
export interface AgentStateEvent {
  state: AgentState;
  message?: string;
}

export interface AgentMessageEvent {
  content: string;
  is_partial: boolean;  // For streaming text
  tool_call?: ToolCall;
}

export interface ProgressEvent {
  step: number;
  total_steps: number;
  file: string;
  progress: number;
  processed: number;
  total: number;
}

export interface PreviewEvent {
  file: string;
  thumbnail: string;  // Base64 or URL
  detections: Detection[];
}

export interface CheckpointEvent {
  reason: 'progress' | 'confidence_drop' | 'error_rate' | 'critical_step';
  stats: ExecutionStats;
  requires_approval: boolean;
}
```

## Implementation Tasks

- [ ] **Tauri IPC Wrapper**
  - [ ] Create `src/lib/utils/tauri.ts`
  - [ ] Implement `invoke<T>()` wrapper with error handling
  - [ ] Add request timeout support
  - [ ] Create typed command functions
  - [ ] Add offline fallback behavior

- [ ] **IPC Command Functions**
  - [ ] `selectFiles()` - Open file picker
  - [ ] `selectDirectory()` - Open folder picker
  - [ ] `getProject(id)` - Load project data
  - [ ] `createProject(name, path)` - Create new project
  - [ ] `startProcessing(request)` - Begin pipeline execution
  - [ ] `pauseProcessing()` - Pause at next safe point
  - [ ] `resumeProcessing()` - Resume from checkpoint
  - [ ] `cancelProcessing()` - Cancel with cleanup
  - [ ] `checkHealth()` - Get system status

- [ ] **SSE Connection Manager**
  - [ ] Create `src/lib/stores/sse.ts`
  - [ ] Implement `EventSource` wrapper
  - [ ] Add automatic reconnection with exponential backoff
  - [ ] Handle connection state (connecting, open, closed, error)
  - [ ] Parse and dispatch events to appropriate stores

- [ ] **SSE Event Handlers**
  - [ ] Route `agent_state` → agentStore.state
  - [ ] Route `agent_message` → agentStore.messages
  - [ ] Route `progress` → executionStore
  - [ ] Route `preview` → executionStore.previews
  - [ ] Route `checkpoint` → trigger checkpoint modal
  - [ ] Route `error` → show error notification

- [ ] **Error Handling**
  - [ ] Create `src/lib/utils/errors.ts`
  - [ ] Define error types (NetworkError, IPCError, ValidationError)
  - [ ] Implement retry logic with max attempts
  - [ ] Add error boundary component
  - [ ] Create toast notifications for errors

- [ ] **Connection Status UI**
  - [ ] Create `ConnectionStatus.svelte` indicator
  - [ ] Show connected/disconnected/reconnecting states
  - [ ] Add manual reconnect button
  - [ ] Display latency indicator

- [ ] **Request Correlation**
  - [ ] Generate correlation IDs for requests
  - [ ] Match responses to pending requests
  - [ ] Implement request timeout handling
  - [ ] Add request cancellation support

## Acceptance Criteria

- [ ] Tauri commands return typed responses
- [ ] SSE reconnects automatically after disconnect
- [ ] UI shows connection status
- [ ] Errors display user-friendly messages
- [ ] Streaming text appears character by character
- [ ] Progress updates are smooth (not jumpy)

## Files to Create/Modify

```
src/lib/
├── types/
│   ├── ipc.ts              # Tauri command types
│   └── sse.ts              # SSE event types
├── stores/
│   └── sse.ts              # SSE connection store
├── utils/
│   ├── tauri.ts            # Tauri IPC wrapper
│   └── errors.ts           # Error types and handlers
└── components/
    └── Layout/
        └── ConnectionStatus.svelte
```

## Tauri IPC Implementation

```typescript
// src/lib/utils/tauri.ts
import { invoke as tauriInvoke } from '@tauri-apps/api/core';

export class IPCError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'IPCError';
  }
}

export async function invoke<T>(
  command: string,
  args?: Record<string, unknown>,
  options?: { timeout?: number }
): Promise<T> {
  const timeout = options?.timeout ?? 30000;

  try {
    const result = await Promise.race([
      tauriInvoke<T>(command, args),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new IPCError('Request timeout', 'TIMEOUT')), timeout)
      ),
    ]);
    return result;
  } catch (error) {
    if (error instanceof IPCError) throw error;
    throw new IPCError(
      error instanceof Error ? error.message : 'Unknown error',
      'IPC_ERROR',
      error
    );
  }
}

// Typed command functions
export const commands = {
  selectFiles: (options?: { multiple?: boolean; filters?: string[] }) =>
    invoke<string[]>('select_files', options),

  selectDirectory: () =>
    invoke<string>('select_directory'),

  checkHealth: () =>
    invoke<HealthStatus>('check_health'),

  startProcessing: (request: ProcessRequest) =>
    invoke<{ job_id: string }>('start_processing', { request }),

  pauseProcessing: (jobId: string) =>
    invoke<void>('pause_processing', { job_id: jobId }),

  resumeProcessing: (jobId: string) =>
    invoke<void>('resume_processing', { job_id: jobId }),

  cancelProcessing: (jobId: string) =>
    invoke<void>('cancel_processing', { job_id: jobId }),
};
```

## SSE Store Implementation

```typescript
// src/lib/stores/sse.ts
import { writable, derived } from 'svelte/store';
import type { SSEEvent, SSEEventType } from '$lib/types/sse';

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

function createSSEStore() {
  const connectionState = writable<ConnectionState>('disconnected');
  const lastError = writable<Error | null>(null);
  const eventSource = writable<EventSource | null>(null);

  let reconnectAttempts = 0;
  const maxReconnectAttempts = 10;
  const baseReconnectDelay = 1000;

  const connect = (url: string, handlers: Record<SSEEventType, (data: unknown) => void>) => {
    connectionState.set('connecting');

    const source = new EventSource(url);

    source.onopen = () => {
      connectionState.set('connected');
      reconnectAttempts = 0;
      lastError.set(null);
    };

    source.onerror = () => {
      connectionState.set('error');
      source.close();
      scheduleReconnect(url, handlers);
    };

    // Register event handlers
    for (const [type, handler] of Object.entries(handlers)) {
      source.addEventListener(type, (event) => {
        try {
          const data = JSON.parse((event as MessageEvent).data);
          handler(data);
        } catch (e) {
          console.error(`Failed to parse SSE event: ${type}`, e);
        }
      });
    }

    eventSource.set(source);
  };

  const scheduleReconnect = (url: string, handlers: Record<SSEEventType, (data: unknown) => void>) => {
    if (reconnectAttempts >= maxReconnectAttempts) {
      lastError.set(new Error('Max reconnection attempts reached'));
      return;
    }

    const delay = Math.min(
      baseReconnectDelay * Math.pow(2, reconnectAttempts),
      30000 // Max 30 seconds
    );

    reconnectAttempts++;
    setTimeout(() => connect(url, handlers), delay);
  };

  const disconnect = () => {
    eventSource.update(source => {
      source?.close();
      return null;
    });
    connectionState.set('disconnected');
  };

  return {
    connectionState,
    lastError,
    connect,
    disconnect,
    isConnected: derived(connectionState, $state => $state === 'connected'),
  };
}

export const sseStore = createSSEStore();
```

## Error Handling Patterns

```typescript
// src/lib/utils/errors.ts

export class NetworkError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'NetworkError';
  }
}

export class ValidationError extends Error {
  constructor(message: string, public field?: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

// User-friendly error messages
export function getErrorMessage(error: unknown): string {
  if (error instanceof NetworkError) {
    if (error.statusCode === 503) {
      return 'Backend service is not available. Please check if the sidecar is running.';
    }
    return `Network error: ${error.message}`;
  }

  if (error instanceof IPCError) {
    switch (error.code) {
      case 'TIMEOUT':
        return 'Request timed out. The operation is taking longer than expected.';
      case 'SIDECAR_NOT_RUNNING':
        return 'Python backend is not running. Please restart the application.';
      default:
        return error.message;
    }
  }

  if (error instanceof Error) {
    return error.message;
  }

  return 'An unexpected error occurred';
}
```

## Notes

- SSE URL will be `http://localhost:8765/stream/{conversation_id}`
- Consider using `@microsoft/fetch-event-source` for better error handling
- Tauri's `invoke` already handles serialization/deserialization
- Add request deduplication for rapid repeated calls
- Consider WebSocket as fallback if SSE has issues
