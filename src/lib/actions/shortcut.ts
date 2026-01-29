/**
 * Svelte action for declarative shortcut registration.
 *
 * Usage:
 * ```svelte
 * <div use:shortcut={{
 *   combo: 'ctrl+k',
 *   action: () => console.log('Triggered!'),
 *   scope: 'global',
 *   description: 'Open command palette'
 * }}>
 * ```
 *
 * Or for sequences:
 * ```svelte
 * <div use:shortcut={{
 *   combo: ['d', 'd'],
 *   action: () => deleteItem(),
 *   scope: 'plan',
 *   description: 'Delete step'
 * }}>
 * ```
 */

import type { Action } from "svelte/action";
import { getKeyboardState } from "$lib/stores/keyboard.svelte";
import type { ShortcutActionOptions } from "$lib/types/keyboard";

/**
 * Svelte action that registers a keyboard shortcut when the element mounts
 * and unregisters it when the element unmounts.
 */
export const shortcut: Action<HTMLElement, ShortcutActionOptions> = (
  _node,
  options
) => {
  const keyboard = getKeyboardState();
  let shortcutId: string | null = null;

  function register(opts: ShortcutActionOptions) {
    // Unregister previous if exists
    if (shortcutId) {
      keyboard.unregister(shortcutId);
    }

    // Register new shortcut
    shortcutId = keyboard.register({
      combo: opts.combo,
      action: opts.action,
      scope: opts.scope ?? "global",
      description: opts.description ?? "",
      category: opts.category,
      priority: opts.priority,
    });
  }

  // Initial registration
  register(options);

  return {
    update(newOptions: ShortcutActionOptions) {
      register(newOptions);
    },
    destroy() {
      if (shortcutId) {
        keyboard.unregister(shortcutId);
      }
    },
  };
};

export default shortcut;
