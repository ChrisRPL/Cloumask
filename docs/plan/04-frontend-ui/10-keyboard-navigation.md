# Keyboard Navigation System

> **Status:** 🔴 Not Started
> **Priority:** P1 (High - accessibility and power-user efficiency)
> **Dependencies:** 01-design-system, 02-core-layout, all view components
> **Estimated Complexity:** Medium

## Overview

Implement a comprehensive keyboard navigation system with vim-like shortcuts, global hotkeys, and context-aware key bindings. Enables power users to navigate the entire application without touching the mouse.

## Goals

- [ ] Global navigation shortcuts (view switching, sidebar)
- [ ] View-specific context shortcuts
- [ ] Vim-like navigation (h/j/k/l)
- [ ] Command palette (Ctrl+K)
- [ ] Shortcut customization
- [ ] Keyboard shortcut help overlay
- [ ] Focus management and trap

## Technical Design

### Shortcut Categories

```typescript
// Shortcut scope definitions
type ShortcutScope =
  | 'global'        // Always active
  | 'chat'          // Active in Chat view
  | 'plan'          // Active in Plan Editor
  | 'execution'     // Active in Execution view
  | 'review'        // Active in Review Queue
  | 'pointcloud'    // Active in Point Cloud viewer
  | 'modal';        // Active when modal is open

interface Shortcut {
  key: string;           // Key combo (e.g., "ctrl+k", "g g")
  description: string;
  scope: ShortcutScope;
  action: () => void;
  enabled?: () => boolean;
}

// Modifier key representation
type Modifier = 'ctrl' | 'alt' | 'shift' | 'meta';
```

### Global Shortcuts Table

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GLOBAL SHORTCUTS                                                       │
├───────────────┬─────────────────────────────────────────────────────────┤
│  Key          │  Action                                                 │
├───────────────┼─────────────────────────────────────────────────────────┤
│  1            │  Go to Chat view                                        │
│  2            │  Go to Plan view                                        │
│  3            │  Go to Execution view                                   │
│  4            │  Go to Review view                                      │
│  5            │  Go to Point Cloud view                                 │
│  , (comma)    │  Go to Settings                                         │
├───────────────┼─────────────────────────────────────────────────────────┤
│  Ctrl+K       │  Open command palette                                   │
│  Ctrl+B       │  Toggle sidebar                                         │
│  Ctrl+/       │  Focus chat input                                       │
│  Ctrl+.       │  Open keyboard shortcuts help                           │
├───────────────┼─────────────────────────────────────────────────────────┤
│  Escape       │  Close modal / Cancel action / Unfocus                  │
│  ?            │  Show keyboard shortcuts overlay                        │
└───────────────┴─────────────────────────────────────────────────────────┘
```

### View-Specific Shortcuts

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CHAT VIEW                                                              │
├───────────────┬─────────────────────────────────────────────────────────┤
│  Enter        │  Send message (when input focused)                      │
│  Shift+Enter  │  New line in message                                    │
│  ↑            │  Edit last message (when input empty)                   │
│  Ctrl+L       │  Clear chat history                                     │
│  Ctrl+E       │  Export chat                                            │
└───────────────┴─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PLAN EDITOR                                                            │
├───────────────┬─────────────────────────────────────────────────────────┤
│  j / ↓        │  Move to next step                                      │
│  k / ↑        │  Move to previous step                                  │
│  Space        │  Toggle step enabled                                    │
│  Enter        │  Open step configuration                                │
│  e            │  Edit selected step                                     │
│  d d          │  Delete selected step                                   │
│  Ctrl+↑       │  Move step up                                           │
│  Ctrl+↓       │  Move step down                                         │
│  n            │  Add new step                                           │
│  Ctrl+Enter   │  Start execution                                        │
│  Escape       │  Close configuration panel                              │
└───────────────┴─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  EXECUTION VIEW                                                         │
├───────────────┬─────────────────────────────────────────────────────────┤
│  Space        │  Pause / Resume execution                               │
│  Escape       │  Cancel execution (with confirmation)                   │
│  r            │  Open Review Queue                                      │
│  Enter        │  Continue past checkpoint                               │
│  e            │  Toggle error log                                       │
└───────────────┴─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  REVIEW QUEUE                                                           │
├───────────────┬─────────────────────────────────────────────────────────┤
│  j / ↓        │  Next item                                              │
│  k / ↑        │  Previous item                                          │
│  a            │  Approve current item                                   │
│  r            │  Reject current item                                    │
│  e            │  Enter edit mode                                        │
│  Delete       │  Delete selected annotation                             │
│  n            │  Add new bounding box                                   │
│  Ctrl+Z       │  Undo                                                   │
│  Ctrl+Y       │  Redo                                                   │
│  Ctrl+Shift+A │  Approve all pending                                    │
│  + / =        │  Zoom in                                                │
│  -            │  Zoom out                                               │
│  0            │  Fit to view                                            │
│  1            │  100% zoom                                              │
│  Escape       │  Exit edit mode / Deselect                              │
└───────────────┴─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  POINT CLOUD VIEWER                                                     │
├───────────────┬─────────────────────────────────────────────────────────┤
│  1            │  Orbit mode                                             │
│  2            │  Pan mode                                               │
│  3            │  First-person mode                                      │
│  r            │  Reset camera                                           │
│  f            │  Focus on selection                                     │
│  m            │  Measurement tool                                       │
│  g            │  Toggle grid                                            │
│  b            │  Toggle bounding boxes                                  │
│  c            │  Cycle color mode                                       │
│  + / =        │  Increase point size                                    │
│  -            │  Decrease point size                                    │
│  h/j/k/l      │  Camera movement (vim-style)                            │
└───────────────┴─────────────────────────────────────────────────────────┘
```

### Component Architecture

```
KeyboardManager (global singleton)
├── ShortcutRegistry
│   ├── register(shortcut: Shortcut)
│   ├── unregister(key: string, scope: ShortcutScope)
│   └── getShortcuts(scope: ShortcutScope)
├── KeyHandler
│   ├── parseKeyCombo(event: KeyboardEvent)
│   ├── matchShortcut(combo: string, scope: ShortcutScope)
│   └── executeAction(shortcut: Shortcut)
├── ScopeManager
│   ├── currentScope: ShortcutScope
│   ├── pushScope(scope: ShortcutScope)
│   └── popScope()
└── FocusManager
    ├── trapFocus(element: HTMLElement)
    ├── releaseFocus()
    └── restoreFocus()

Components:
├── KeyboardShortcutsOverlay.svelte
├── CommandPalette.svelte
└── ShortcutHint.svelte
```

## Implementation Tasks

- [ ] **Keyboard Manager**
  - [ ] Create `src/lib/utils/keyboard.ts`
  - [ ] Implement key combo parsing
  - [ ] Handle modifier keys (Ctrl, Alt, Shift, Meta)
  - [ ] Support key sequences (e.g., "g g", "d d")
  - [ ] Debounce rapid key presses
  - [ ] Prevent browser default shortcuts when needed

- [ ] **Shortcut Registry**
  - [ ] Store shortcuts by scope
  - [ ] Dynamic registration/unregistration
  - [ ] Conflict detection
  - [ ] Priority handling for overlapping shortcuts

- [ ] **Scope Management**
  - [ ] Track current active scope
  - [ ] Scope stack for modals/dialogs
  - [ ] Automatic scope switching on view change
  - [ ] Parent scope fallback

- [ ] **Focus Management**
  - [ ] Focus trap for modals
  - [ ] Skip links for accessibility
  - [ ] Focus restoration after modal close
  - [ ] Visible focus indicators

- [ ] **KeyboardShortcutsOverlay Component**
  - [ ] Create `KeyboardShortcutsOverlay.svelte`
  - [ ] Grid display of shortcuts by scope
  - [ ] Search/filter shortcuts
  - [ ] Show currently active scope
  - [ ] Trigger with `?` key

- [ ] **CommandPalette Component**
  - [ ] Create `CommandPalette.svelte`
  - [ ] Fuzzy search for commands
  - [ ] Recent commands history
  - [ ] Keyboard navigation (↑/↓/Enter)
  - [ ] Show shortcut hints

- [ ] **ShortcutHint Component**
  - [ ] Create `ShortcutHint.svelte`
  - [ ] Small badge showing key combo
  - [ ] Tooltip with description
  - [ ] Conditional display based on platform

- [ ] **Settings Integration**
  - [ ] Shortcut customization UI
  - [ ] Import/export shortcuts
  - [ ] Reset to defaults
  - [ ] Persist to localStorage/settings

- [ ] **Vim-like Navigation**
  - [ ] h/j/k/l for directional movement
  - [ ] g g for go-to-top
  - [ ] G for go-to-bottom
  - [ ] / for search
  - [ ] n/N for next/previous match

## Acceptance Criteria

- [ ] All views navigable via keyboard only
- [ ] Shortcuts shown in UI hints/tooltips
- [ ] `?` key shows shortcut overlay
- [ ] `Ctrl+K` opens command palette
- [ ] Focus is managed correctly in modals
- [ ] No conflicts with browser shortcuts
- [ ] Shortcuts work on all major browsers

## Files to Create/Modify

```
src/lib/
├── utils/
│   └── keyboard.ts          # Keyboard manager
├── stores/
│   └── keyboard.ts          # Keyboard state store
├── components/
│   ├── KeyboardShortcutsOverlay.svelte
│   ├── CommandPalette.svelte
│   ├── ShortcutHint.svelte
│   └── FocusTrap.svelte
└── actions/
    └── shortcut.ts          # Svelte action for shortcuts
```

## Keyboard Manager Implementation

```typescript
// src/lib/utils/keyboard.ts

type KeyCombo = {
  key: string;
  ctrl: boolean;
  alt: boolean;
  shift: boolean;
  meta: boolean;
};

interface ShortcutHandler {
  combo: string;
  scope: ShortcutScope;
  description: string;
  action: () => void;
  enabled?: () => boolean;
}

class KeyboardManager {
  private shortcuts: Map<string, ShortcutHandler[]> = new Map();
  private scopeStack: ShortcutScope[] = ['global'];
  private sequenceBuffer: string[] = [];
  private sequenceTimeout: number | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', this.handleKeyDown.bind(this));
    }
  }

  get currentScope(): ShortcutScope {
    return this.scopeStack[this.scopeStack.length - 1];
  }

  pushScope(scope: ShortcutScope): void {
    this.scopeStack.push(scope);
  }

  popScope(): void {
    if (this.scopeStack.length > 1) {
      this.scopeStack.pop();
    }
  }

  register(handler: ShortcutHandler): () => void {
    const key = this.normalizeCombo(handler.combo);
    if (!this.shortcuts.has(key)) {
      this.shortcuts.set(key, []);
    }
    this.shortcuts.get(key)!.push(handler);

    // Return unregister function
    return () => {
      const handlers = this.shortcuts.get(key);
      if (handlers) {
        const index = handlers.indexOf(handler);
        if (index > -1) handlers.splice(index, 1);
      }
    };
  }

  private handleKeyDown(event: KeyboardEvent): void {
    // Ignore if typing in input
    if (this.isTypingInInput(event)) return;

    const combo = this.parseKeyEvent(event);

    // Handle key sequences (e.g., "g g")
    if (this.isSequenceStart(combo)) {
      this.sequenceBuffer.push(combo);
      this.sequenceTimeout = window.setTimeout(() => {
        this.sequenceBuffer = [];
      }, 500);
      return;
    }

    const fullCombo = [...this.sequenceBuffer, combo].join(' ');
    this.sequenceBuffer = [];

    if (this.sequenceTimeout) {
      clearTimeout(this.sequenceTimeout);
      this.sequenceTimeout = null;
    }

    // Find and execute matching shortcut
    const handlers = this.shortcuts.get(fullCombo) || [];
    for (const handler of handlers) {
      if (this.isInScope(handler.scope) && (!handler.enabled || handler.enabled())) {
        event.preventDefault();
        handler.action();
        return;
      }
    }
  }

  private parseKeyEvent(event: KeyboardEvent): string {
    const parts: string[] = [];
    if (event.ctrlKey || event.metaKey) parts.push('ctrl');
    if (event.altKey) parts.push('alt');
    if (event.shiftKey) parts.push('shift');
    parts.push(event.key.toLowerCase());
    return parts.join('+');
  }

  private normalizeCombo(combo: string): string {
    return combo.toLowerCase().split('+').sort().join('+');
  }

  private isInScope(scope: ShortcutScope): boolean {
    return scope === 'global' || scope === this.currentScope;
  }

  private isTypingInInput(event: KeyboardEvent): boolean {
    const target = event.target as HTMLElement;
    const tagName = target.tagName.toLowerCase();
    const isEditable = target.isContentEditable;
    return ['input', 'textarea', 'select'].includes(tagName) || isEditable;
  }

  private isSequenceStart(combo: string): boolean {
    // Check if this combo starts a multi-key sequence
    for (const key of this.shortcuts.keys()) {
      if (key.startsWith(combo + ' ')) return true;
    }
    return false;
  }
}

export const keyboardManager = new KeyboardManager();
```

## Svelte Action for Shortcuts

```typescript
// src/lib/actions/shortcut.ts
import { keyboardManager } from '$lib/utils/keyboard';

interface ShortcutParams {
  combo: string;
  scope?: ShortcutScope;
  description?: string;
  action: () => void;
}

export function shortcut(node: HTMLElement, params: ShortcutParams) {
  const unregister = keyboardManager.register({
    combo: params.combo,
    scope: params.scope || 'global',
    description: params.description || '',
    action: params.action,
  });

  return {
    update(newParams: ShortcutParams) {
      unregister();
      keyboardManager.register({
        combo: newParams.combo,
        scope: newParams.scope || 'global',
        description: newParams.description || '',
        action: newParams.action,
      });
    },
    destroy() {
      unregister();
    },
  };
}
```

## Command Palette

```svelte
<!-- CommandPalette.svelte -->
<script lang="ts">
  import { keyboardManager } from '$lib/utils/keyboard';
  import { Dialog, DialogContent } from '$lib/components/ui/dialog';

  let open = $state(false);
  let query = $state('');
  let selectedIndex = $state(0);

  const commands = $derived(
    keyboardManager.getAllShortcuts()
      .filter(s => s.description.toLowerCase().includes(query.toLowerCase()))
  );

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'ArrowDown') {
      selectedIndex = Math.min(selectedIndex + 1, commands.length - 1);
    } else if (event.key === 'ArrowUp') {
      selectedIndex = Math.max(selectedIndex - 1, 0);
    } else if (event.key === 'Enter' && commands[selectedIndex]) {
      commands[selectedIndex].action();
      open = false;
    }
  }

  // Register Ctrl+K to open
  $effect(() => {
    return keyboardManager.register({
      combo: 'ctrl+k',
      scope: 'global',
      description: 'Open command palette',
      action: () => { open = true; query = ''; selectedIndex = 0; },
    });
  });
</script>

<Dialog bind:open>
  <DialogContent class="max-w-lg">
    <input
      type="text"
      bind:value={query}
      on:keydown={handleKeyDown}
      placeholder="Type a command..."
      class="w-full p-3 text-lg bg-transparent border-b border-border"
    />
    <ul class="max-h-80 overflow-y-auto">
      {#each commands as command, i}
        <li
          class="px-3 py-2 cursor-pointer"
          class:bg-secondary={i === selectedIndex}
          on:click={() => { command.action(); open = false; }}
        >
          <span class="font-medium">{command.description}</span>
          <span class="ml-auto text-muted-foreground">{command.combo}</span>
        </li>
      {/each}
    </ul>
  </DialogContent>
</Dialog>
```

## Accessibility Considerations

- [ ] All interactive elements focusable via Tab
- [ ] Visible focus indicators with high contrast
- [ ] Skip links at page top
- [ ] Focus trap in modals
- [ ] Announce shortcuts to screen readers
- [ ] Support reduced motion preference

## Notes

- Mac users: Meta key (⌘) maps to Ctrl for shortcuts
- Consider adding "hold to preview shortcut" feature
- Test with assistive technologies (VoiceOver, NVDA)
- Document shortcuts in help/documentation
- Consider gamepad/controller support for navigation
