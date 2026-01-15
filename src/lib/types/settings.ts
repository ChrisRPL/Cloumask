/**
 * Settings type definitions for user preferences and application configuration.
 */

// ============================================================================
// Theme
// ============================================================================

export type Theme = "dark" | "light" | "system";

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

export interface KeyboardShortcuts {
  toggleSidebar: string;
  nextView: string;
  previousView: string;
  focusChat: string;
  approveStep: string;
  rejectStep: string;
}

// ============================================================================
// Model Defaults
// ============================================================================

export interface ModelDefaults {
  detection: string;
  segmentation: string;
  anonymization: string;
  confidence: number;
}

// ============================================================================
// Settings
// ============================================================================

export interface Settings {
  theme: Theme;
  keyboardShortcuts: KeyboardShortcuts;
  modelDefaults: ModelDefaults;
  autoSaveInterval: number;
  showConfidenceScores: boolean;
  enableSoundEffects: boolean;
  language: string;
}

// ============================================================================
// Settings State Interface
// ============================================================================

export interface SettingsState {
  readonly settings: Settings;

  // Actions
  updateSetting<K extends keyof Settings>(key: K, value: Settings[K]): void;
  updateShortcut<K extends keyof KeyboardShortcuts>(
    key: K,
    value: string,
  ): void;
  updateModelDefault<K extends keyof ModelDefaults>(
    key: K,
    value: ModelDefaults[K],
  ): void;
  resetToDefaults(): void;
  exportSettings(): string;
  importSettings(json: string): boolean;
}
