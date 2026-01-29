/**
 * Keyboard navigation system types.
 * Provides type definitions for shortcuts, scopes, and the keyboard state.
 */

/**
 * Keyboard scope determines when shortcuts are active.
 * Scopes are hierarchical - 'global' shortcuts always apply,
 * while view-specific scopes only apply when that view is active.
 */
export type KeyboardScope =
  | "global" // Always active
  | "chat" // Chat view specific
  | "plan" // Plan editor specific
  | "execution" // Execution view specific
  | "review" // Review queue specific
  | "pointcloud" // Point cloud viewer specific
  | "modal" // Modal/dialog context
  | "command-palette"; // Command palette open

/**
 * Parsed key combination structure.
 * Represents a single key press with modifiers.
 */
export interface KeyCombo {
  /** The main key (lowercase, e.g., 'k', 'enter', 'escape') */
  key: string;
  /** Ctrl key pressed (or Cmd on Mac for platform-normalized shortcuts) */
  ctrl: boolean;
  /** Alt/Option key pressed */
  alt: boolean;
  /** Shift key pressed */
  shift: boolean;
  /** Meta/Cmd key pressed (raw, not normalized) */
  meta: boolean;
}

/**
 * Shortcut category for grouping in the help overlay.
 */
export type ShortcutCategory =
  | "Navigation"
  | "Plan Editor"
  | "Review"
  | "Execution"
  | "Point Cloud"
  | "Chat"
  | "Help";

/**
 * Registered shortcut definition.
 */
export interface Shortcut {
  /** Unique identifier for the shortcut */
  id: string;
  /** Key combination string (e.g., 'ctrl+k') or sequence array (e.g., ['d', 'd']) */
  combo: string | string[];
  /** Function to execute when shortcut is triggered */
  action: () => void;
  /** Scope in which this shortcut is active */
  scope: KeyboardScope;
  /** Human-readable description for help overlay */
  description: string;
  /** Category for grouping in help overlay */
  category?: ShortcutCategory;
  /** Priority for conflict resolution (higher wins, default 0) */
  priority?: number;
  /** Whether the shortcut is currently enabled */
  enabled?: boolean;
}

/**
 * Command for the command palette.
 * Extends shortcut with search-related metadata.
 */
export interface Command {
  /** Unique identifier */
  id: string;
  /** Display label in command palette */
  label: string;
  /** Optional description shown below label */
  description?: string;
  /** Shortcut string to display (e.g., '⌘K') */
  shortcut?: string;
  /** Function to execute when selected */
  action: () => void;
  /** Category for grouping */
  category?: ShortcutCategory;
  /** Additional keywords for fuzzy search */
  keywords?: string[];
  /** Lucide icon name */
  icon?: string;
}

/**
 * Shortcut conflict information.
 */
export interface ShortcutConflict {
  /** ID of the existing shortcut */
  existingId: string;
  /** Description of the existing shortcut */
  existingDescription: string;
  /** ID of the new conflicting shortcut */
  newId: string;
  /** The conflicting key combination */
  combo: string;
}

/**
 * Platform type for display purposes.
 */
export type Platform = "mac" | "windows" | "linux";

/**
 * Keyboard state interface.
 * Defines the public API for the keyboard store.
 */
export interface KeyboardState {
  // Getters
  readonly activeScope: KeyboardScope;
  readonly scopeStack: readonly KeyboardScope[];
  readonly registeredShortcuts: ReadonlyMap<string, Shortcut>;
  readonly isCommandPaletteOpen: boolean;
  readonly isHelpOverlayOpen: boolean;
  readonly pendingSequence: readonly KeyCombo[];
  readonly lastKeyTime: number;
  readonly shortcutsByCategory: ReadonlyMap<ShortcutCategory, Shortcut[]>;
  readonly platform: Platform;

  // Scope management
  pushScope(scope: KeyboardScope): void;
  popScope(): void;
  setScope(scope: KeyboardScope): void;

  // Shortcut registration
  register(shortcut: Omit<Shortcut, "id">): string;
  unregister(id: string): void;
  setEnabled(id: string, enabled: boolean): void;

  // Command palette
  openCommandPalette(): void;
  closeCommandPalette(): void;
  toggleCommandPalette(): void;

  // Help overlay
  openHelpOverlay(): void;
  closeHelpOverlay(): void;
  toggleHelpOverlay(): void;

  // Query helpers
  getShortcutsForScope(scope: KeyboardScope): Shortcut[];
  getAllCommands(): Command[];
  checkConflict(combo: string, scope: KeyboardScope): ShortcutConflict | null;

  // Internal (called by layout)
  handleKeyEvent(event: KeyboardEvent): boolean;
}

/**
 * Shortcut action options for the use:shortcut directive.
 */
export interface ShortcutActionOptions {
  /** Key combination string or sequence array */
  combo: string | string[];
  /** Function to execute */
  action: () => void;
  /** Scope for the shortcut (default: 'global') */
  scope?: KeyboardScope;
  /** Description for help overlay */
  description?: string;
  /** Category for grouping */
  category?: ShortcutCategory;
  /** Priority for conflict resolution */
  priority?: number;
}
