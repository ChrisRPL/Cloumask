/**
 * Pipeline type definitions for the plan editor and step configuration.
 *
 * These types mirror the Python backend types in backend/src/backend/agent/state.py
 * to ensure type-safe communication between frontend and backend.
 */

// ============================================================================
// Step Types
// ============================================================================

export type StepType =
  | "detection"
  | "segmentation"
  | "anonymization"
  | "export"
  | "classification"
  | "custom";

export type StepStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "skipped";

// ============================================================================
// Step Configuration
// ============================================================================

export interface StepConfig {
  model?: string;
  confidence?: number;
  batchSize?: number;
  params: Record<string, unknown>;
}

// ============================================================================
// Pipeline Step
// ============================================================================

export interface PipelineStep {
  id: string;
  toolName: string;
  type: StepType;
  description: string;
  config: StepConfig;
  status: StepStatus;
  order: number;
  critical?: boolean;
  result?: Record<string, unknown>;
  error?: string;
  startedAt?: string;
  completedAt?: string;
}

// ============================================================================
// Pipeline State Interface
// ============================================================================

export interface PipelineState {
  readonly steps: PipelineStep[];
  readonly isEditing: boolean;
  readonly selectedStepId: string | null;
  readonly isDirty: boolean;
  readonly pipelineId: string | null;

  // Derived
  readonly sortedSteps: PipelineStep[];
  readonly enabledSteps: PipelineStep[];
  readonly completedSteps: PipelineStep[];
  readonly hasFailedSteps: boolean;
  readonly stepCount: number;

  // CRUD Actions
  addStep(step: Omit<PipelineStep, "id" | "order" | "status">): PipelineStep;
  updateStep(id: string, updates: Partial<PipelineStep>): void;
  removeStep(id: string): void;

  // Reordering
  moveStep(fromIndex: number, toIndex: number): void;

  // Selection
  selectStep(id: string | null): void;

  // State management
  setSteps(steps: PipelineStep[]): void;
  setPipelineId(id: string | null): void;
  setEditing(editing: boolean): void;
  clearPipeline(): void;
  markClean(): void;
  reset(): void;
}
