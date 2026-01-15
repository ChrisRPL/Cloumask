/**
 * Execution state management using Svelte 5 runes and context.
 *
 * Provides centralized state for execution progress tracking, statistics,
 * error management, and checkpoint handling during pipeline execution.
 */

import { getContext, setContext } from "svelte";
import type {
  ExecutionState,
  ExecutionStatus,
  ExecutionProgress,
  ExecutionStats,
  ExecutionError,
  CheckpointInfo,
  CheckpointTrigger,
  QualityMetrics,
} from "$lib/types/execution";

// Re-export types for convenience
export type {
  ExecutionState,
  ExecutionStatus,
  ExecutionProgress,
  ExecutionStats,
  ExecutionError,
  CheckpointInfo,
  CheckpointTrigger,
  QualityMetrics,
};

// ============================================================================
// Constants
// ============================================================================

const EXECUTION_STATE_KEY = Symbol("execution-state");

const INITIAL_STATS: ExecutionStats = {
  processed: 0,
  detected: 0,
  flagged: 0,
  errors: 0,
  startedAt: null,
  estimatedCompletion: null,
};

const INITIAL_PROGRESS: ExecutionProgress = {
  current: 0,
  total: 0,
  percentage: 0,
};

// ============================================================================
// Helpers
// ============================================================================

function now(): string {
  return new Date().toISOString();
}

function calculatePercentage(current: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((current / total) * 100);
}

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates execution state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createExecutionState(): ExecutionState {
  // Reactive state
  let status = $state<ExecutionStatus>("idle");
  let progress = $state<ExecutionProgress>({ ...INITIAL_PROGRESS });
  let stats = $state<ExecutionStats>({ ...INITIAL_STATS });
  let errors = $state<ExecutionError[]>([]);
  let currentStepId = $state<string | null>(null);
  let checkpoint = $state<CheckpointInfo | null>(null);

  // Derived values
  const isRunning = $derived(status === "running");
  const isPaused = $derived(status === "paused");
  const isComplete = $derived(status === "completed" || status === "failed");
  const hasErrors = $derived(errors.length > 0);
  const errorRate = $derived(
    stats.processed > 0 ? stats.errors / stats.processed : 0,
  );

  return {
    // Getters
    get status() {
      return status;
    },
    get progress() {
      return progress;
    },
    get stats() {
      return stats;
    },
    get errors() {
      return errors;
    },
    get currentStepId() {
      return currentStepId;
    },
    get checkpoint() {
      return checkpoint;
    },

    // Derived getters
    get isRunning() {
      return isRunning;
    },
    get isPaused() {
      return isPaused;
    },
    get isComplete() {
      return isComplete;
    },
    get hasErrors() {
      return hasErrors;
    },
    get errorRate() {
      return errorRate;
    },

    // Status actions
    setStatus(newStatus: ExecutionStatus) {
      status = newStatus;
    },

    start() {
      status = "running";
      stats = {
        ...INITIAL_STATS,
        startedAt: now(),
      };
      progress = { ...INITIAL_PROGRESS };
      errors = [];
      checkpoint = null;
    },

    pause() {
      if (status === "running") {
        status = "paused";
      }
    },

    resume() {
      if (status === "paused" || status === "checkpoint") {
        status = "running";
        checkpoint = null;
      }
    },

    cancel() {
      status = "cancelled";
      checkpoint = null;
    },

    reset() {
      status = "idle";
      progress = { ...INITIAL_PROGRESS };
      stats = { ...INITIAL_STATS };
      errors = [];
      currentStepId = null;
      checkpoint = null;
    },

    // Progress actions
    updateProgress(current: number, total: number) {
      progress = {
        current,
        total,
        percentage: calculatePercentage(current, total),
      };

      // Update ETA based on progress
      if (stats.startedAt && current > 0) {
        const elapsed = Date.now() - new Date(stats.startedAt).getTime();
        const msPerItem = elapsed / current;
        const remaining = (total - current) * msPerItem;
        stats = {
          ...stats,
          estimatedCompletion: new Date(Date.now() + remaining).toISOString(),
        };
      }
    },

    setCurrentStep(stepId: string | null) {
      currentStepId = stepId;
    },

    // Stats actions
    updateStats(newStats: Partial<ExecutionStats>) {
      stats = { ...stats, ...newStats };
    },

    incrementProcessed() {
      stats = { ...stats, processed: stats.processed + 1 };
    },

    incrementDetected(count = 1) {
      stats = { ...stats, detected: stats.detected + count };
    },

    incrementFlagged(count = 1) {
      stats = { ...stats, flagged: stats.flagged + count };
    },

    // Error actions
    addError(error: Omit<ExecutionError, "timestamp">) {
      const newError: ExecutionError = {
        ...error,
        timestamp: now(),
      };
      errors = [...errors, newError];
      stats = { ...stats, errors: stats.errors + 1 };
    },

    clearErrors() {
      errors = [];
    },

    // Checkpoint actions
    setCheckpoint(newCheckpoint: CheckpointInfo | null) {
      checkpoint = newCheckpoint;
      if (newCheckpoint) {
        status = "checkpoint";
      }
    },

    clearCheckpoint() {
      checkpoint = null;
      if (status === "checkpoint") {
        status = "running";
      }
    },
  };
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize execution state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setExecutionState(): ExecutionState {
  const state = createExecutionState();
  setContext(EXECUTION_STATE_KEY, state);
  return state;
}

/**
 * Get execution state from Svelte context.
 * Call this in child components that need execution state.
 */
export function getExecutionState(): ExecutionState {
  return getContext<ExecutionState>(EXECUTION_STATE_KEY);
}
