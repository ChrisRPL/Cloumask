/**
 * Execution type definitions for progress tracking and checkpoint management.
 *
 * These types handle real-time updates from the backend via SSE events.
 */

// ============================================================================
// Execution Status
// ============================================================================

export type ExecutionStatus =
  | "idle"
  | "running"
  | "paused"
  | "checkpoint"
  | "completed"
  | "failed"
  | "cancelled";

// ============================================================================
// Progress Tracking
// ============================================================================

export interface ExecutionProgress {
  current: number;
  total: number;
  percentage: number;
}

// ============================================================================
// Execution Statistics
// ============================================================================

export interface ExecutionStats {
  processed: number;
  detected: number;
  flagged: number;
  errors: number;
  startedAt: string | null;
  estimatedCompletion: string | null;
}

export interface PreviewAnnotation {
  label: string;
  confidence: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface PointcloudPreviewAnnotation {
  id: string;
  className: string;
  confidence: number;
  center: [number, number, number];
  size: [number, number, number];
  yaw: number;
  status: "pending" | "accepted" | "rejected" | "edited";
}

export type PreviewAssetType = "image" | "pointcloud";

export interface PreviewItem {
  id: string;
  imagePath: string;
  thumbnailUrl: string;
  annotations: PreviewAnnotation[];
  assetType?: PreviewAssetType;
  pointcloudAnnotations?: PointcloudPreviewAnnotation[];
  status: "processed" | "flagged" | "error";
}

// ============================================================================
// Execution Errors
// ============================================================================

export interface ExecutionError {
  stepId: string;
  message: string;
  timestamp: string;
  recoverable: boolean;
}

// ============================================================================
// Checkpoint Types
// ============================================================================

export type CheckpointTrigger =
  | "percentage"
  | "quality_drop"
  | "error_rate"
  | "critical_step";

export interface QualityMetrics {
  averageConfidence: number;
  errorCount: number;
  totalProcessed: number;
  processingSpeed?: number;
}

export interface CheckpointInfo {
  id: string;
  stepIndex: number;
  triggerReason: CheckpointTrigger;
  progressPercent: number;
  qualityMetrics: QualityMetrics;
  message: string;
  createdAt: string;
}

// ============================================================================
// Execution State Interface
// ============================================================================

export interface ExecutionState {
  readonly status: ExecutionStatus;
  readonly progress: ExecutionProgress;
  readonly stats: ExecutionStats;
  readonly previews: PreviewItem[];
  readonly selectedPointcloudPreview: PreviewItem | null;
  readonly errors: ExecutionError[];
  readonly currentStepId: string | null;
  readonly checkpoint: CheckpointInfo | null;

  // Derived
  readonly isRunning: boolean;
  readonly isPaused: boolean;
  readonly isComplete: boolean;
  readonly hasErrors: boolean;
  readonly errorRate: number;

  // Status actions
  setStatus(status: ExecutionStatus): void;
  start(): void;
  pause(): void;
  resume(): void;
  cancel(): void;
  reset(): void;

  // Progress actions
  updateProgress(current: number, total: number): void;
  setCurrentStep(stepId: string | null): void;

  // Stats actions
  updateStats(stats: Partial<ExecutionStats>): void;
  incrementProcessed(): void;
  incrementDetected(count?: number): void;
  incrementFlagged(count?: number): void;
  setPreviews(previews: PreviewItem[]): void;
  appendPreviews(previews: PreviewItem[]): void;
  clearPreviews(): void;
  setSelectedPointcloudPreview(preview: PreviewItem | null): void;

  // Error actions
  addError(error: Omit<ExecutionError, "timestamp">): void;
  clearErrors(): void;

  // Checkpoint actions
  setCheckpoint(checkpoint: CheckpointInfo | null): void;
  clearCheckpoint(): void;
}
