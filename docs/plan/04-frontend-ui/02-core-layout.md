# Core Layout & Navigation

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical - provides app shell for all views)
> **Dependencies:** 01-design-system
> **Estimated Complexity:** Medium

## Overview

Implement the main application shell with sidebar navigation, header bar, and content area. This layout hosts all views (Chat, Plan, Execution, Review, Point Cloud) and provides consistent navigation structure.

## Goals

- [ ] App shell with sidebar + main content layout
- [ ] Collapsible sidebar navigation
- [ ] Project selector in header
- [ ] Window controls integration (Tauri)
- [ ] Responsive layout for different window sizes
- [ ] View routing and transitions

## Technical Design

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HEADER (h-12, fixed)                                                   │
│  ┌────────┐                                              ┌────────────┐ │
│  │ Logo   │  Cloumask                    [Project ▼]     │ [_][□][X]  │ │
│  └────────┘                                              └────────────┘ │
├─────────────┬───────────────────────────────────────────────────────────┤
│  SIDEBAR    │  MAIN CONTENT                                             │
│  (w-16/64)  │                                                           │
│  ┌─────────┐│  ┌───────────────────────────────────────────────────────┐│
│  │ 💬 Chat ││  │                                                       ││
│  │ 📋 Plan ││  │   [View content rendered here]                        ││
│  │ ▶ Exec  ││  │                                                       ││
│  │ ✓ Review││  │                                                       ││
│  │ ☁ 3D    ││  │                                                       ││
│  │─────────││  │                                                       ││
│  │ ⚙ Set.  ││  │                                                       ││
│  └─────────┘│  └───────────────────────────────────────────────────────┘│
└─────────────┴───────────────────────────────────────────────────────────┘
```

### Sidebar States

```
Collapsed (w-16):           Expanded (w-64):
┌────────────────┐          ┌────────────────────────────┐
│   [≡]          │          │ [≡]  Cloumask              │
│                │          │                            │
│   💬           │          │ 💬  Chat                   │
│   📋           │          │ 📋  Plan                   │
│   ▶            │          │ ▶   Execute                │
│   ✓            │          │ ✓   Review                 │
│   ☁            │          │ ☁   Point Cloud            │
│                │          │                            │
│   ⚙            │          │ ⚙   Settings               │
└────────────────┘          └────────────────────────────┘
```

### Route Structure

```typescript
type View = 'chat' | 'plan' | 'execute' | 'review' | 'pointcloud' | 'settings';

// Routes map to view components
const routes: Record<View, Component> = {
  chat: ChatPanel,
  plan: PlanEditor,
  execute: ExecutionView,
  review: ReviewQueue,
  pointcloud: PointCloudViewer,
  settings: SettingsPanel,
};
```

### Component Hierarchy

```
+layout.svelte
├── Header.svelte
│   ├── Logo.svelte
│   ├── ProjectSelector.svelte
│   └── WindowControls.svelte
├── Sidebar.svelte
│   ├── SidebarToggle.svelte
│   ├── NavItem.svelte (× 6)
│   └── SidebarFooter.svelte
└── MainContent.svelte
    └── [Dynamic view component]
```

## Implementation Tasks

- [ ] **App Shell Structure**
  - [ ] Create `+layout.svelte` with grid layout
  - [ ] Implement CSS Grid: `grid-template-columns: auto 1fr`
  - [ ] Add `grid-template-rows: auto 1fr` for header/content
  - [ ] Set `min-height: 100vh` and `overflow: hidden`

- [ ] **Header Component**
  - [ ] Create `Header.svelte` with fixed positioning
  - [ ] Add app logo (SVG, 32x32)
  - [ ] Add app title "Cloumask"
  - [ ] Implement `ProjectSelector.svelte` dropdown
  - [ ] Add Tauri window controls (minimize, maximize, close)
  - [ ] Style with `backdrop-blur` for glass effect

- [ ] **Sidebar Navigation**
  - [ ] Create `Sidebar.svelte` container
  - [ ] Implement collapse/expand animation (w-16 ↔ w-64)
  - [ ] Create `NavItem.svelte` with icon + label + active state
  - [ ] Add keyboard shortcut hints in expanded mode
  - [ ] Persist sidebar state in localStorage
  - [ ] Add separator between main nav and settings

- [ ] **Navigation Items**
  - [ ] Chat - `MessageSquare` icon, shortcut `1`
  - [ ] Plan - `ClipboardList` icon, shortcut `2`
  - [ ] Execute - `Play` icon, shortcut `3`
  - [ ] Review - `CheckSquare` icon, shortcut `4`
  - [ ] Point Cloud - `Cloud` icon, shortcut `5`
  - [ ] Settings - `Settings` icon, shortcut `,`

- [ ] **Main Content Area**
  - [ ] Create `MainContent.svelte` wrapper
  - [ ] Implement view switching based on current route
  - [ ] Add page transition animations (fade/slide)
  - [ ] Handle scroll areas per-view
  - [ ] Add loading skeleton during view transitions

- [ ] **Responsive Behavior**
  - [ ] Auto-collapse sidebar below 1024px
  - [ ] Allow manual toggle at any size
  - [ ] Mobile: overlay sidebar with backdrop
  - [ ] Remember user preference per breakpoint

- [ ] **Window Controls (Tauri)**
  - [ ] Create `WindowControls.svelte`
  - [ ] Implement minimize via `appWindow.minimize()`
  - [ ] Implement maximize/restore toggle
  - [ ] Implement close via `appWindow.close()`
  - [ ] Add hover effects matching OS style

## Acceptance Criteria

- [ ] App loads with sidebar and header visible
- [ ] Clicking nav items switches main content
- [ ] Sidebar collapses/expands with smooth animation
- [ ] Window controls work (minimize, maximize, close)
- [ ] Layout adapts to window resize
- [ ] Number keys 1-5 switch between views
- [ ] Active nav item is visually highlighted
- [ ] Project selector shows current project name

## Files to Create/Modify

```
src/
├── routes/
│   ├── +layout.svelte          # Main app shell
│   ├── +page.svelte            # Default view (Chat)
│   └── +error.svelte           # Error boundary
└── lib/
    └── components/
        └── Layout/
            ├── Header.svelte
            ├── Logo.svelte
            ├── ProjectSelector.svelte
            ├── WindowControls.svelte
            ├── Sidebar.svelte
            ├── SidebarToggle.svelte
            ├── NavItem.svelte
            ├── MainContent.svelte
            └── index.ts
```

## State Management

```typescript
// src/lib/stores/ui.ts
import { writable, derived } from 'svelte/store';

export type View = 'chat' | 'plan' | 'execute' | 'review' | 'pointcloud' | 'settings';

// Current active view
export const currentView = writable<View>('chat');

// Sidebar state
export const sidebarExpanded = writable<boolean>(true);

// Derived: sidebar width class
export const sidebarWidth = derived(
  sidebarExpanded,
  $expanded => $expanded ? 'w-64' : 'w-16'
);
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Go to Chat |
| `2` | Go to Plan |
| `3` | Go to Execute |
| `4` | Go to Review |
| `5` | Go to Point Cloud |
| `,` | Go to Settings |
| `[` | Collapse sidebar |
| `]` | Expand sidebar |
| `Ctrl+B` | Toggle sidebar |

## Animations

```css
/* Sidebar collapse/expand */
.sidebar {
  transition: width 200ms ease-out;
}

/* View transitions */
.view-enter {
  animation: fadeSlideIn 200ms ease-out;
}

@keyframes fadeSlideIn {
  from {
    opacity: 0;
    transform: translateX(10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}
```

## Notes

- Use `data-tauri-drag-region` attribute for draggable header area
- Window controls should only render on desktop (check `window.__TAURI__`)
- Consider preloading adjacent views for faster navigation
- Logo should be crisp at all DPI levels (use SVG)
