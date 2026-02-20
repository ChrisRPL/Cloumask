/**
 * Settings state management using Svelte 5 runes and context.
 *
 * Provides centralized state for user preferences with localStorage persistence.
 */

import { getContext, setContext } from "svelte";
import type {
  Settings,
  SettingsState,
  Theme,
  KeyboardShortcuts,
  ModelDefaults,
} from "$lib/types/settings";

// Re-export types for convenience
export type {
  Settings,
  SettingsState,
  Theme,
  KeyboardShortcuts,
  ModelDefaults,
};

// ============================================================================
// Constants
// ============================================================================

const SETTINGS_STATE_KEY = Symbol("settings-state");
const STORAGE_KEY = "cloumask:settings";

export const DEFAULT_SETTINGS: Settings = {
  theme: "light",
  keyboardShortcuts: {
    toggleSidebar: "Ctrl+B",
    nextView: "Ctrl+]",
    previousView: "Ctrl+[",
    focusChat: "Ctrl+/",
    approveStep: "Ctrl+Enter",
    rejectStep: "Ctrl+Backspace",
  },
  modelDefaults: {
    detection: "yolo11m",
    segmentation: "sam2",
    anonymization: "blur",
    confidence: 0.5,
  },
  autoSaveInterval: 30000,
  showConfidenceScores: true,
  enableSoundEffects: false,
  language: "en",
};

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates settings state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createSettingsState(): SettingsState {
  // Load persisted settings
  const getInitialSettings = (): Settings => {
    if (typeof window === "undefined") return DEFAULT_SETTINGS;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return DEFAULT_SETTINGS;
    try {
      const parsed = JSON.parse(stored) as Partial<Settings>;
      // Deep merge with defaults to handle new settings added in updates
      return {
        ...DEFAULT_SETTINGS,
        ...parsed,
        keyboardShortcuts: {
          ...DEFAULT_SETTINGS.keyboardShortcuts,
          ...parsed.keyboardShortcuts,
        },
        modelDefaults: {
          ...DEFAULT_SETTINGS.modelDefaults,
          ...parsed.modelDefaults,
        },
      };
    } catch {
      return DEFAULT_SETTINGS;
    }
  };

  // Reactive state using Svelte 5 runes
  let settings = $state<Settings>(getInitialSettings());

  // Persist settings on change (debounced via effect)
  $effect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }
  });

  return {
    // Getters
    get settings() {
      return settings;
    },

    // Actions
    updateSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
      settings = { ...settings, [key]: value };
    },

    updateShortcut<K extends keyof KeyboardShortcuts>(key: K, value: string) {
      settings = {
        ...settings,
        keyboardShortcuts: {
          ...settings.keyboardShortcuts,
          [key]: value,
        },
      };
    },

    updateModelDefault<K extends keyof ModelDefaults>(
      key: K,
      value: ModelDefaults[K],
    ) {
      settings = {
        ...settings,
        modelDefaults: {
          ...settings.modelDefaults,
          [key]: value,
        },
      };
    },

    resetToDefaults() {
      settings = { ...DEFAULT_SETTINGS };
    },

    exportSettings(): string {
      return JSON.stringify(settings, null, 2);
    },

    importSettings(json: string): boolean {
      try {
        const imported = JSON.parse(json) as Partial<Settings>;
        settings = {
          ...DEFAULT_SETTINGS,
          ...imported,
          keyboardShortcuts: {
            ...DEFAULT_SETTINGS.keyboardShortcuts,
            ...imported.keyboardShortcuts,
          },
          modelDefaults: {
            ...DEFAULT_SETTINGS.modelDefaults,
            ...imported.modelDefaults,
          },
        };
        return true;
      } catch {
        return false;
      }
    },
  };
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize settings state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setSettingsState(): SettingsState {
  const state = createSettingsState();
  setContext(SETTINGS_STATE_KEY, state);
  return state;
}

/**
 * Get settings state from Svelte context.
 * Call this in child components that need settings.
 */
export function getSettingsState(): SettingsState {
  return getContext<SettingsState>(SETTINGS_STATE_KEY);
}
