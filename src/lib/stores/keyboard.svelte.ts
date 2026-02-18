/**
 * Keyboard navigation state management using Svelte 5 runes and context.
 *
 * Provides centralized state for keyboard shortcuts, scope management,
 * command palette, and help overlay.
 */

import { getContext, setContext } from "svelte";
import type {
  KeyboardScope,
  KeyboardState,
  Shortcut,
  ShortcutCategory,
  Command,
  ShortcutConflict,
  KeyCombo,
  Platform,
} from "$lib/types/keyboard";
import {
  getPlatform,
  formatComboString,
  isInputElement,
  generateShortcutId,
  normalizeCombo,
  isSequenceStartCandidate,
  eventToCombo,
  comboToString,
  SEQUENCE_TIMEOUT,
} from "$lib/utils/keyboard";

// Re-export types for convenience
export type { KeyboardScope, KeyboardState, Shortcut, Command };

// ============================================================================
// Constants
// ============================================================================

const KEYBOARD_STATE_KEY = Symbol("keyboard-state");

/** Default shortcut category order for display */
export const CATEGORY_ORDER: ShortcutCategory[] = [
  "Navigation",
  "Chat",
  "Plan Editor",
  "Execution",
  "Review",
  "Point Cloud",
  "Help",
];

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates keyboard state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createKeyboardState(): KeyboardState {
  // Detect platform once at initialization
  const platform: Platform =
    typeof window !== "undefined" ? getPlatform() : "linux";

  // Reactive state using Svelte 5 runes
  let activeScope = $state<KeyboardScope>("global");
  let scopeStack = $state<KeyboardScope[]>(["global"]);
  let registeredShortcuts = $state(new Map<string, Shortcut>());
  let isCommandPaletteOpen = $state(false);
  let isHelpOverlayOpen = $state(false);
  let pendingSequence = $state<KeyCombo[]>([]);
  let lastKeyTime = $state(0);
  let sequenceTimeoutId: ReturnType<typeof setTimeout> | null = null;

  // Derived: shortcuts grouped by category for help overlay
  const shortcutsByCategory = $derived.by(() => {
    const grouped = new Map<ShortcutCategory, Shortcut[]>();

    for (const shortcut of registeredShortcuts.values()) {
      if (shortcut.enabled === false) continue;

      const category = shortcut.category ?? "Navigation";
      const list = grouped.get(category) ?? [];
      list.push(shortcut);
      grouped.set(category, list);
    }

    return grouped;
  });

  // Helper: clear pending sequence
  function clearSequence() {
    pendingSequence = [];
    if (sequenceTimeoutId) {
      clearTimeout(sequenceTimeoutId);
      sequenceTimeoutId = null;
    }
  }

  // Helper: check if scope is active
  function isScopeActive(scope: KeyboardScope): boolean {
    return scope === "global" || scope === activeScope;
  }

  // Helper: find matching shortcut for a combo
  function findMatchingShortcut(combo: string): Shortcut | null {
    const normalized = normalizeCombo(combo);
    let bestMatch: Shortcut | null = null;
    let bestPriority = -Infinity;

    for (const shortcut of registeredShortcuts.values()) {
      if (shortcut.enabled === false) continue;
      if (!isScopeActive(shortcut.scope)) continue;

      // Handle string combo
      if (typeof shortcut.combo === "string") {
        if (normalizeCombo(shortcut.combo) === normalized) {
          const priority = shortcut.priority ?? 0;
          if (priority > bestPriority) {
            bestMatch = shortcut;
            bestPriority = priority;
          }
        }
      }
    }

    return bestMatch;
  }

  // Helper: find matching sequence shortcut
  function findMatchingSequence(sequence: KeyCombo[]): Shortcut | null {
    const sequenceStr = sequence.map((c) => comboToString(c));

    for (const shortcut of registeredShortcuts.values()) {
      if (shortcut.enabled === false) continue;
      if (!isScopeActive(shortcut.scope)) continue;

      // Handle array combo (sequence)
      if (Array.isArray(shortcut.combo)) {
        if (shortcut.combo.length !== sequenceStr.length) continue;

        const matches = shortcut.combo.every(
          (c, i) => normalizeCombo(c) === normalizeCombo(sequenceStr[i])
        );

        if (matches) {
          return shortcut;
        }
      }
    }

    return null;
  }

  // Helper: check if sequence could match (partial)
  function couldMatchSequence(sequence: KeyCombo[]): boolean {
    const sequenceStr = sequence.map((c) => comboToString(c));

    for (const shortcut of registeredShortcuts.values()) {
      if (shortcut.enabled === false) continue;
      if (!isScopeActive(shortcut.scope)) continue;

      if (Array.isArray(shortcut.combo)) {
        if (shortcut.combo.length <= sequenceStr.length) continue;

        const matches = sequenceStr.every(
          (s, i) => normalizeCombo(s) === normalizeCombo(shortcut.combo[i])
        );

        if (matches) {
          return true;
        }
      }
    }

    return false;
  }

  return {
    // Getters (reactive via closure)
    get activeScope() {
      return activeScope;
    },
    get scopeStack() {
      return scopeStack;
    },
    get registeredShortcuts() {
      return registeredShortcuts;
    },
    get isCommandPaletteOpen() {
      return isCommandPaletteOpen;
    },
    get isHelpOverlayOpen() {
      return isHelpOverlayOpen;
    },
    get pendingSequence() {
      return pendingSequence;
    },
    get lastKeyTime() {
      return lastKeyTime;
    },
    get shortcutsByCategory() {
      return shortcutsByCategory;
    },
    get platform() {
      return platform;
    },

    // Scope management
    pushScope(scope: KeyboardScope) {
      scopeStack = [...scopeStack, scope];
      activeScope = scope;
    },

    popScope() {
      if (scopeStack.length > 1) {
        scopeStack = scopeStack.slice(0, -1);
        activeScope = scopeStack[scopeStack.length - 1];
      }
    },

    setScope(scope: KeyboardScope) {
      // Replace current scope without pushing (for view changes)
      if (scopeStack.length > 1) {
        scopeStack = [...scopeStack.slice(0, -1), scope];
      } else {
        scopeStack = ["global", scope];
      }
      activeScope = scope;
    },

    // Shortcut registration
    register(shortcut: Omit<Shortcut, "id">): string {
      const id = generateShortcutId();
      const newShortcut: Shortcut = {
        ...shortcut,
        id,
        enabled: shortcut.enabled ?? true,
      };

      // Create new Map to trigger reactivity
      const newMap = new Map(registeredShortcuts);
      newMap.set(id, newShortcut);
      registeredShortcuts = newMap;

      return id;
    },

    unregister(id: string) {
      if (registeredShortcuts.has(id)) {
        const newMap = new Map(registeredShortcuts);
        newMap.delete(id);
        registeredShortcuts = newMap;
      }
    },

    setEnabled(id: string, enabled: boolean) {
      const shortcut = registeredShortcuts.get(id);
      if (shortcut) {
        const newMap = new Map(registeredShortcuts);
        newMap.set(id, { ...shortcut, enabled });
        registeredShortcuts = newMap;
      }
    },

    // Command palette
    openCommandPalette() {
      isCommandPaletteOpen = true;
      this.pushScope("command-palette");
    },

    closeCommandPalette() {
      isCommandPaletteOpen = false;
      this.popScope();
    },

    toggleCommandPalette() {
      if (isCommandPaletteOpen) {
        this.closeCommandPalette();
      } else {
        this.openCommandPalette();
      }
    },

    // Help overlay
    openHelpOverlay() {
      isHelpOverlayOpen = true;
    },

    closeHelpOverlay() {
      isHelpOverlayOpen = false;
    },

    toggleHelpOverlay() {
      isHelpOverlayOpen = !isHelpOverlayOpen;
    },

    // Query helpers
    getShortcutsForScope(scope: KeyboardScope): Shortcut[] {
      return Array.from(registeredShortcuts.values()).filter(
        (s) =>
          s.enabled !== false && (s.scope === scope || s.scope === "global")
      );
    },

    getAllCommands(): Command[] {
      const commands: Command[] = [];

      for (const shortcut of registeredShortcuts.values()) {
        if (shortcut.enabled === false) continue;

        const comboDisplay =
          typeof shortcut.combo === "string"
            ? formatComboString(shortcut.combo, platform)
            : shortcut.combo
                .map((c) => formatComboString(c, platform))
                .join(" ");

        commands.push({
          id: shortcut.id,
          label: shortcut.description,
          shortcut: comboDisplay,
          action: shortcut.action,
          category: shortcut.category,
        });
      }

      return commands;
    },

    checkConflict(
      combo: string,
      scope: KeyboardScope
    ): ShortcutConflict | null {
      const normalized = normalizeCombo(combo);

      for (const shortcut of registeredShortcuts.values()) {
        if (shortcut.enabled === false) continue;

        // Check if scopes overlap
        const scopesOverlap =
          shortcut.scope === "global" ||
          scope === "global" ||
          shortcut.scope === scope;

        if (!scopesOverlap) continue;

        if (typeof shortcut.combo === "string") {
          if (normalizeCombo(shortcut.combo) === normalized) {
            return {
              existingId: shortcut.id,
              existingDescription: shortcut.description,
              newId: "",
              combo,
            };
          }
        }
      }

      return null;
    },

    // Main event handler (called by layout)
    handleKeyEvent(event: KeyboardEvent): boolean {
      // Skip if in input element (unless it's Escape)
      if (isInputElement(event.target) && event.key !== "Escape") {
        return false;
      }

      const now = Date.now();
      const combo = eventToCombo(event);
      const comboStr = comboToString(combo);

      // Check for sequence timeout
      if (now - lastKeyTime > SEQUENCE_TIMEOUT) {
        clearSequence();
      }

      lastKeyTime = now;

      // Handle sequence candidates (single chars without modifiers)
      if (isSequenceStartCandidate(combo)) {
        const newSequence = [...pendingSequence, combo];

        // Check if we have a complete sequence match
        const sequenceMatch = findMatchingSequence(newSequence);
        if (sequenceMatch) {
          event.preventDefault();
          clearSequence();
          sequenceMatch.action();
          return true;
        }

        // Check if this could be part of a longer sequence
        if (couldMatchSequence(newSequence)) {
          pendingSequence = newSequence;

          // Set timeout to clear sequence
          if (sequenceTimeoutId) {
            clearTimeout(sequenceTimeoutId);
          }
          sequenceTimeoutId = setTimeout(() => {
            clearSequence();
          }, SEQUENCE_TIMEOUT);

          // Don't prevent default yet - might not be a sequence
          return false;
        }
      }

      // Clear any pending sequence since this isn't a sequence key
      clearSequence();

      // Look for single combo match
      const match = findMatchingShortcut(comboStr);
      if (match) {
        event.preventDefault();
        match.action();
        return true;
      }

      return false;
    },
  };
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize keyboard state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setKeyboardState(): KeyboardState {
  const state = createKeyboardState();
  setContext(KEYBOARD_STATE_KEY, state);
  return state;
}

/**
 * Get keyboard state from Svelte context.
 * Call this in child components that need keyboard state.
 */
export function getKeyboardState(): KeyboardState {
  return getContext<KeyboardState>(KEYBOARD_STATE_KEY);
}
