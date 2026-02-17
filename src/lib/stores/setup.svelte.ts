/**
 * Setup state management using Svelte 5 runes and context.
 *
 * Tracks first-time setup progress and completion status.
 * Persists to localStorage so setup only runs once.
 */

import { getContext, setContext } from "svelte";
import { isTauri } from "$lib/utils/tauri";

// ============================================================================
// Types
// ============================================================================

export type SetupStep =
  | "prerequisites"
  | "download_llm"
  | "build_executor"
  | "complete";

export interface SetupProgress {
  /** Current step in the setup process. */
  currentStep: SetupStep;
  /** Progress percentage for current step (0-100). */
  stepProgress: number;
  /** Human-readable status message. */
  statusMessage: string;
  /** Whether an error occurred. */
  hasError: boolean;
  /** Error message if any. */
  errorMessage: string | null;
}

export interface SetupState {
  /** Whether setup has been completed. */
  readonly isComplete: boolean;
  /** Whether setup is currently in progress. */
  readonly isInProgress: boolean;
  /** Current setup progress. */
  readonly progress: SetupProgress;

  /** Start the setup process. */
  startSetup(): void;
  /** Update progress for current step. */
  updateProgress(progress: number, message: string): void;
  /** Move to next step. */
  nextStep(): void;
  /** Report an error. */
  setError(message: string): void;
  /** Clear error and retry current step. */
  retry(): void;
  /** Mark setup as complete. */
  markComplete(): void;
  /** Reset setup (for debugging/testing). */
  reset(): void;
}

// ============================================================================
// Constants
// ============================================================================

const SETUP_STATE_KEY = Symbol("setup-state");
const STORAGE_KEY = "cloumask:setup";

const STEP_ORDER: SetupStep[] = [
  "prerequisites",
  "download_llm",
  "build_executor",
  "complete",
];

const STEP_MESSAGES: Record<SetupStep, string> = {
  prerequisites: "Checking prerequisites...",
  download_llm: "Downloading AI model...",
  build_executor: "Building script executor...",
  complete: "Setup complete!",
};

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates setup state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createSetupState(): SetupState {
  // Check if setup was previously completed
  const getInitialComplete = (): boolean => {
    if (typeof window === "undefined") return false;
    // Desktop should always enter the initialization screen on app launch.
    // It verifies that sidecar and model bootstrap are started before the main UI.
    if (isTauri()) return false;
    return localStorage.getItem(STORAGE_KEY) === "complete";
  };

  // Reactive state using Svelte 5 runes
  let isComplete = $state<boolean>(getInitialComplete());
  let isInProgress = $state<boolean>(false);
  let progress = $state<SetupProgress>({
    currentStep: "prerequisites",
    stepProgress: 0,
    statusMessage: "",
    hasError: false,
    errorMessage: null,
  });

  // Persist completion status
  $effect(() => {
    if (typeof window !== "undefined" && isComplete) {
      localStorage.setItem(STORAGE_KEY, "complete");
    }
  });

  return {
    // Getters
    get isComplete() {
      return isComplete;
    },
    get isInProgress() {
      return isInProgress;
    },
    get progress() {
      return progress;
    },

    // Actions
    startSetup() {
      isInProgress = true;
      progress = {
        currentStep: "prerequisites",
        stepProgress: 0,
        statusMessage: STEP_MESSAGES.prerequisites,
        hasError: false,
        errorMessage: null,
      };
    },

    updateProgress(stepProgress: number, message: string) {
      progress = {
        ...progress,
        stepProgress,
        statusMessage: message,
      };
    },

    nextStep() {
      const currentIndex = STEP_ORDER.indexOf(progress.currentStep);
      if (currentIndex < STEP_ORDER.length - 1) {
        const nextStep = STEP_ORDER[currentIndex + 1];
        progress = {
          currentStep: nextStep,
          stepProgress: 0,
          statusMessage: STEP_MESSAGES[nextStep],
          hasError: false,
          errorMessage: null,
        };
      }
    },

    setError(message: string) {
      progress = {
        ...progress,
        hasError: true,
        errorMessage: message,
      };
    },

    retry() {
      progress = {
        ...progress,
        stepProgress: 0,
        hasError: false,
        errorMessage: null,
        statusMessage: STEP_MESSAGES[progress.currentStep],
      };
    },

    markComplete() {
      isComplete = true;
      isInProgress = false;
      progress = {
        currentStep: "complete",
        stepProgress: 100,
        statusMessage: STEP_MESSAGES.complete,
        hasError: false,
        errorMessage: null,
      };
    },

    reset() {
      if (typeof window !== "undefined") {
        localStorage.removeItem(STORAGE_KEY);
      }
      isComplete = false;
      isInProgress = false;
      progress = {
        currentStep: "prerequisites",
        stepProgress: 0,
        statusMessage: "",
        hasError: false,
        errorMessage: null,
      };
    },
  };
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize setup state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setSetupState(): SetupState {
  const state = createSetupState();
  setContext(SETUP_STATE_KEY, state);
  return state;
}

/**
 * Get setup state from Svelte context.
 * Call this in child components that need setup state.
 */
export function getSetupState(): SetupState {
  return getContext<SetupState>(SETUP_STATE_KEY);
}
