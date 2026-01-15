/**
 * Pipeline state management using Svelte 5 runes and context.
 *
 * Provides centralized state for the plan editor including pipeline steps,
 * CRUD operations, drag-drop reordering, and step configuration.
 */

import { getContext, setContext } from "svelte";
import type {
  PipelineStep,
  PipelineState,
  StepType,
  StepStatus,
  StepConfig,
} from "$lib/types/pipeline";

// Re-export types for convenience
export type { PipelineStep, PipelineState, StepType, StepStatus, StepConfig };

// ============================================================================
// Constants
// ============================================================================

const PIPELINE_STATE_KEY = Symbol("pipeline-state");

// ============================================================================
// Helpers
// ============================================================================

function generateId(): string {
  return crypto.randomUUID();
}

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates pipeline state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createPipelineState(): PipelineState {
  // Reactive state
  let steps = $state<PipelineStep[]>([]);
  let isEditing = $state(false);
  let selectedStepId = $state<string | null>(null);
  let isDirty = $state(false);
  let pipelineId = $state<string | null>(null);

  // Derived values
  const sortedSteps = $derived([...steps].sort((a, b) => a.order - b.order));
  const enabledSteps = $derived(steps.filter((s) => s.status !== "skipped"));
  const completedSteps = $derived(
    steps.filter((s) => s.status === "completed"),
  );
  const hasFailedSteps = $derived(steps.some((s) => s.status === "failed"));
  const stepCount = $derived(steps.length);

  return {
    // Getters
    get steps() {
      return steps;
    },
    get isEditing() {
      return isEditing;
    },
    get selectedStepId() {
      return selectedStepId;
    },
    get isDirty() {
      return isDirty;
    },
    get pipelineId() {
      return pipelineId;
    },

    // Derived getters
    get sortedSteps() {
      return sortedSteps;
    },
    get enabledSteps() {
      return enabledSteps;
    },
    get completedSteps() {
      return completedSteps;
    },
    get hasFailedSteps() {
      return hasFailedSteps;
    },
    get stepCount() {
      return stepCount;
    },

    // CRUD Actions
    addStep(step: Omit<PipelineStep, "id" | "order" | "status">): PipelineStep {
      const newStep: PipelineStep = {
        ...step,
        id: generateId(),
        order: steps.length,
        status: "pending",
      };
      steps = [...steps, newStep];
      isDirty = true;
      return newStep;
    },

    updateStep(id: string, updates: Partial<PipelineStep>) {
      steps = steps.map((step) =>
        step.id === id ? { ...step, ...updates } : step,
      );
      isDirty = true;
    },

    removeStep(id: string) {
      const removedIndex = steps.findIndex((s) => s.id === id);
      if (removedIndex === -1) return;

      // Remove step and reorder remaining
      steps = steps
        .filter((s) => s.id !== id)
        .map((s, index) => ({
          ...s,
          order: index,
        }));

      // Clear selection if removed step was selected
      if (selectedStepId === id) {
        selectedStepId = null;
      }
      isDirty = true;
    },

    // Reordering
    moveStep(fromIndex: number, toIndex: number) {
      if (fromIndex === toIndex) return;
      if (fromIndex < 0 || fromIndex >= steps.length) return;
      if (toIndex < 0 || toIndex >= steps.length) return;

      const sorted = [...sortedSteps];
      const [moved] = sorted.splice(fromIndex, 1);
      sorted.splice(toIndex, 0, moved);

      // Update order for all steps
      steps = sorted.map((step, index) => ({
        ...step,
        order: index,
      }));
      isDirty = true;
    },

    // Selection
    selectStep(id: string | null) {
      selectedStepId = id;
    },

    // State management
    setSteps(newSteps: PipelineStep[]) {
      steps = newSteps.map((step, index) => ({
        ...step,
        order: step.order ?? index,
      }));
      isDirty = false;
    },

    setPipelineId(id: string | null) {
      pipelineId = id;
    },

    setEditing(editing: boolean) {
      isEditing = editing;
    },

    clearPipeline() {
      steps = [];
      selectedStepId = null;
      pipelineId = null;
      isDirty = false;
    },

    markClean() {
      isDirty = false;
    },

    reset() {
      steps = [];
      isEditing = false;
      selectedStepId = null;
      isDirty = false;
      pipelineId = null;
    },
  };
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize pipeline state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setPipelineState(): PipelineState {
  const state = createPipelineState();
  setContext(PIPELINE_STATE_KEY, state);
  return state;
}

/**
 * Get pipeline state from Svelte context.
 * Call this in child components that need pipeline state.
 */
export function getPipelineState(): PipelineState {
  return getContext<PipelineState>(PIPELINE_STATE_KEY);
}
