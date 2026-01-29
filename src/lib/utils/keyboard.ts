/**
 * Keyboard utility functions.
 * Pure, stateless functions for key parsing, matching, and display formatting.
 */

import type { KeyCombo, Platform } from "$lib/types/keyboard";

/** Sequence timeout in milliseconds (for vim-style key sequences) */
export const SEQUENCE_TIMEOUT = 500;

/** Special key name mappings for display */
const KEY_DISPLAY_NAMES: Record<string, string> = {
  " ": "Space",
  arrowup: "↑",
  arrowdown: "↓",
  arrowleft: "←",
  arrowright: "→",
  enter: "↵",
  escape: "Esc",
  backspace: "⌫",
  delete: "Del",
  tab: "Tab",
  capslock: "Caps",
};

/** Mac-specific modifier symbols */
const MAC_MODIFIERS: Record<string, string> = {
  ctrl: "⌘", // On Mac, we map Ctrl shortcuts to Cmd
  alt: "⌥",
  shift: "⇧",
  meta: "⌘",
};

/** Windows/Linux modifier display */
const STANDARD_MODIFIERS: Record<string, string> = {
  ctrl: "Ctrl",
  alt: "Alt",
  shift: "Shift",
  meta: "Win",
};

/**
 * Detect the current platform.
 */
export function getPlatform(): Platform {
  if (typeof navigator === "undefined") return "linux";

  const platform = navigator.platform?.toLowerCase() || "";
  const userAgent = navigator.userAgent?.toLowerCase() || "";

  if (platform.includes("mac") || userAgent.includes("mac")) {
    return "mac";
  }
  if (platform.includes("win") || userAgent.includes("win")) {
    return "windows";
  }
  return "linux";
}

/**
 * Parse a key combo string into a KeyCombo object.
 * Handles formats like: 'ctrl+k', 'Ctrl+K', 'cmd+shift+p', 'escape', 'enter'
 */
export function parseCombo(combo: string): KeyCombo {
  const parts = combo.toLowerCase().split("+").map((p) => p.trim());

  const result: KeyCombo = {
    key: "",
    ctrl: false,
    alt: false,
    shift: false,
    meta: false,
  };

  for (const part of parts) {
    switch (part) {
      case "ctrl":
      case "control":
      case "cmd":
      case "command":
        result.ctrl = true;
        break;
      case "alt":
      case "option":
        result.alt = true;
        break;
      case "shift":
        result.shift = true;
        break;
      case "meta":
      case "win":
      case "super":
        result.meta = true;
        break;
      default:
        // This is the actual key
        result.key = part;
    }
  }

  return result;
}

/**
 * Format a KeyCombo for display.
 * Returns platform-appropriate symbols (e.g., '⌘K' on Mac, 'Ctrl+K' on Windows).
 */
export function formatCombo(combo: KeyCombo, platform?: Platform): string {
  const p = platform ?? getPlatform();
  const modifiers = p === "mac" ? MAC_MODIFIERS : STANDARD_MODIFIERS;
  const separator = p === "mac" ? "" : "+";

  const parts: string[] = [];

  if (combo.ctrl) parts.push(modifiers.ctrl);
  if (combo.alt) parts.push(modifiers.alt);
  if (combo.shift) parts.push(modifiers.shift);
  if (combo.meta && p !== "mac") parts.push(modifiers.meta);

  // Format the key
  const keyDisplay =
    KEY_DISPLAY_NAMES[combo.key] || combo.key.toUpperCase();
  parts.push(keyDisplay);

  return parts.join(separator);
}

/**
 * Format a combo string for display (convenience wrapper).
 */
export function formatComboString(combo: string, platform?: Platform): string {
  return formatCombo(parseCombo(combo), platform);
}

/**
 * Format a key sequence for display.
 * E.g., ['g', 'g'] becomes 'g g' or ['d', 'd'] becomes 'd d'
 */
export function formatSequence(sequence: string[], platform?: Platform): string {
  return sequence
    .map((key) => formatComboString(key, platform))
    .join(" ");
}

/**
 * Check if a KeyboardEvent matches a KeyCombo.
 * Handles platform normalization (Cmd on Mac = Ctrl shortcuts).
 */
export function matchesCombo(event: KeyboardEvent, combo: KeyCombo): boolean {
  const platform = getPlatform();

  // Normalize key
  const eventKey = event.key.toLowerCase();

  // Key must match
  if (eventKey !== combo.key) {
    // Handle special cases
    if (combo.key === "escape" && eventKey !== "escape") return false;
    if (combo.key === "enter" && eventKey !== "enter") return false;
    if (combo.key === " " && eventKey !== " ") return false;
    if (combo.key !== "escape" && combo.key !== "enter" && combo.key !== " ") {
      return false;
    }
  }

  // On Mac, Cmd (metaKey) is used for shortcuts that are Ctrl on other platforms
  const ctrlMatch =
    platform === "mac"
      ? (event.metaKey === combo.ctrl) // On Mac, Cmd = Ctrl for shortcuts
      : (event.ctrlKey === combo.ctrl);

  // Alt/Option
  const altMatch = event.altKey === combo.alt;

  // Shift
  const shiftMatch = event.shiftKey === combo.shift;

  // Meta is only checked on non-Mac for explicit meta shortcuts
  const metaMatch =
    platform === "mac" ? true : event.metaKey === combo.meta;

  return ctrlMatch && altMatch && shiftMatch && metaMatch;
}

/**
 * Check if an event target is an input element where shortcuts should be ignored.
 */
export function isInputElement(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) return false;

  const tagName = target.tagName.toLowerCase();

  // Check for standard input elements
  if (["input", "textarea", "select"].includes(tagName)) {
    return true;
  }

  // Check for contenteditable
  if (target.isContentEditable) {
    return true;
  }

  // Check for role="textbox" (accessibility pattern)
  if (target.getAttribute("role") === "textbox") {
    return true;
  }

  return false;
}

/**
 * Generate a unique shortcut ID.
 */
export function generateShortcutId(): string {
  return `shortcut_${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Normalize a combo string for comparison.
 * Sorts modifiers alphabetically and lowercases everything.
 */
export function normalizeCombo(combo: string): string {
  const parsed = parseCombo(combo);
  const parts: string[] = [];

  // Add modifiers in consistent order
  if (parsed.alt) parts.push("alt");
  if (parsed.ctrl) parts.push("ctrl");
  if (parsed.meta) parts.push("meta");
  if (parsed.shift) parts.push("shift");

  // Add the key
  parts.push(parsed.key);

  return parts.join("+");
}

/**
 * Check if a combo is a potential sequence start.
 * Single-character combos without modifiers can start sequences.
 */
export function isSequenceStartCandidate(combo: KeyCombo): boolean {
  return (
    !combo.ctrl &&
    !combo.alt &&
    !combo.shift &&
    !combo.meta &&
    combo.key.length === 1
  );
}

/**
 * Create a KeyCombo from a KeyboardEvent.
 */
export function eventToCombo(event: KeyboardEvent): KeyCombo {
  const platform = getPlatform();

  return {
    key: event.key.toLowerCase(),
    // On Mac, normalize Cmd to ctrl for our internal representation
    ctrl: platform === "mac" ? event.metaKey : event.ctrlKey,
    alt: event.altKey,
    shift: event.shiftKey,
    meta: platform === "mac" ? false : event.metaKey,
  };
}

/**
 * Convert a KeyCombo to a normalized string.
 */
export function comboToString(combo: KeyCombo): string {
  const parts: string[] = [];

  if (combo.ctrl) parts.push("ctrl");
  if (combo.alt) parts.push("alt");
  if (combo.shift) parts.push("shift");
  if (combo.meta) parts.push("meta");
  parts.push(combo.key);

  return parts.join("+");
}

/**
 * Simple fuzzy search scoring function.
 * Returns a score (higher is better match), or 0 if no match.
 */
export function fuzzyScore(query: string, target: string): number {
  if (!query) return 1; // Empty query matches everything

  const q = query.toLowerCase();
  const t = target.toLowerCase();

  // Exact match
  if (t === q) return 100;

  // Starts with
  if (t.startsWith(q)) return 80;

  // Contains
  if (t.includes(q)) return 60;

  // Fuzzy character matching
  let score = 0;
  let queryIdx = 0;

  for (let i = 0; i < t.length && queryIdx < q.length; i++) {
    if (t[i] === q[queryIdx]) {
      score += 1;
      queryIdx++;
    }
  }

  // Only return score if all query characters were matched
  return queryIdx === q.length ? score : 0;
}
