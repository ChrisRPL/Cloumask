/**
 * LocalStorage utilities with SSR safety.
 *
 * Provides type-safe access to localStorage with proper handling
 * for server-side rendering where window is undefined.
 */

// ============================================================================
// Storage Adapter
// ============================================================================

export interface StorageAdapter<T> {
  get(): T;
  set(value: T): void;
  remove(): void;
  exists(): boolean;
}

/**
 * Creates a type-safe localStorage adapter.
 *
 * @param key - The localStorage key
 * @param defaultValue - Default value when key doesn't exist
 * @returns Storage adapter with get/set/remove methods
 *
 * @example
 * ```typescript
 * const themeStorage = createStorageAdapter('cloumask:theme', 'light');
 * const theme = themeStorage.get(); // 'light' if not set
 * themeStorage.set('dark');
 * ```
 */
export function createStorageAdapter<T>(
  key: string,
  defaultValue: T,
): StorageAdapter<T> {
  return {
    get(): T {
      if (typeof window === "undefined") return defaultValue;
      const stored = localStorage.getItem(key);
      if (stored === null) return defaultValue;
      try {
        return JSON.parse(stored) as T;
      } catch {
        return defaultValue;
      }
    },

    set(value: T): void {
      if (typeof window === "undefined") return;
      localStorage.setItem(key, JSON.stringify(value));
    },

    remove(): void {
      if (typeof window === "undefined") return;
      localStorage.removeItem(key);
    },

    exists(): boolean {
      if (typeof window === "undefined") return false;
      return localStorage.getItem(key) !== null;
    },
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if localStorage is available.
 */
export function isStorageAvailable(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const testKey = "__storage_test__";
    localStorage.setItem(testKey, testKey);
    localStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get a value from localStorage with a fallback.
 */
export function getStorageItem<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  const stored = localStorage.getItem(key);
  if (stored === null) return fallback;
  try {
    return JSON.parse(stored) as T;
  } catch {
    return fallback;
  }
}

/**
 * Set a value in localStorage.
 */
export function setStorageItem<T>(key: string, value: T): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(key, JSON.stringify(value));
}

/**
 * Remove a value from localStorage.
 */
export function removeStorageItem(key: string): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(key);
}

// ============================================================================
// Storage Keys
// ============================================================================

/**
 * Centralized storage key definitions for consistency.
 */
export const STORAGE_KEYS = {
  // UI state
  SIDEBAR_EXPANDED: "cloumask:sidebar:expanded",
  CURRENT_VIEW: "cloumask:view:current",

  // Settings
  SETTINGS: "cloumask:settings",
  THEME: "cloumask:theme",

  // Agent
  DRAFT_MESSAGE: "cloumask:agent:draft",

  // Pipeline
  PIPELINE_DRAFT: "cloumask:pipeline:draft",
} as const;
