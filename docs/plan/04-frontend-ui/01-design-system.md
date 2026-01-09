# Design System Setup

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical - blocks all other frontend work)
> **Dependencies:** 01-foundation (Tauri + Svelte project scaffolded)
> **Estimated Complexity:** Medium

## Overview

Configure the foundational design system: Tailwind CSS with dark-mode-first configuration, shadcn/ui components via bits-ui, Lucide icons, and theme tokens. This spec establishes the visual foundation for all UI components.

## Goals

- [ ] Tailwind CSS configured with custom dark theme
- [ ] shadcn/ui initialized with bits-ui adapter
- [ ] Design tokens defined (colors, spacing, typography)
- [ ] Base UI components generated (Button, Card, Input, etc.)
- [ ] Lucide icons integrated
- [ ] CSS variables for theme switching support

## Technical Design

### Color Palette (Dark Mode Default)

```css
/* Base colors */
--background: 240 10% 4%;        /* #0a0a0b */
--foreground: 0 0% 98%;          /* #fafafa */

/* Card/Panel backgrounds */
--card: 240 6% 10%;              /* #18181b */
--card-foreground: 0 0% 98%;

/* Primary accent (Violet) */
--primary: 263 70% 50%;          /* #8b5cf6 */
--primary-foreground: 0 0% 98%;

/* Secondary */
--secondary: 240 4% 16%;         /* #27272a */
--secondary-foreground: 0 0% 98%;

/* Muted */
--muted: 240 4% 16%;
--muted-foreground: 240 5% 65%;

/* Semantic colors */
--destructive: 0 84% 60%;        /* #ef4444 - errors */
--success: 142 71% 45%;          /* #22c55e - success */
--warning: 38 92% 50%;           /* #f59e0b - warnings */

/* Borders & inputs */
--border: 240 4% 16%;
--input: 240 4% 16%;
--ring: 263 70% 50%;             /* Focus ring - violet */

/* Radius */
--radius: 0.5rem;
```

### Typography Scale

```css
/* Font family */
--font-sans: 'Inter', system-ui, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;

/* Size scale (rem) */
--text-xs: 0.75rem;    /* 12px */
--text-sm: 0.875rem;   /* 14px */
--text-base: 1rem;     /* 16px */
--text-lg: 1.125rem;   /* 18px */
--text-xl: 1.25rem;    /* 20px */
--text-2xl: 1.5rem;    /* 24px */
--text-3xl: 1.875rem;  /* 30px */
```

### Spacing Scale

```css
/* Using Tailwind's default spacing with additions */
--space-px: 1px;
--space-0.5: 0.125rem;  /* 2px */
--space-1: 0.25rem;     /* 4px */
--space-2: 0.5rem;      /* 8px */
--space-3: 0.75rem;     /* 12px */
--space-4: 1rem;        /* 16px */
--space-6: 1.5rem;      /* 24px */
--space-8: 2rem;        /* 32px */
--space-12: 3rem;       /* 48px */
--space-16: 4rem;       /* 64px */
```

### Component Architecture

```
src/lib/components/ui/
├── button.svelte          # Primary, secondary, ghost, destructive variants
├── card.svelte            # Card, CardHeader, CardContent, CardFooter
├── input.svelte           # Text input with validation states
├── textarea.svelte        # Multi-line input
├── select.svelte          # Dropdown select
├── checkbox.svelte        # Checkbox with label
├── switch.svelte          # Toggle switch
├── badge.svelte           # Status badges
├── avatar.svelte          # User/agent avatars
├── tooltip.svelte         # Hover tooltips
├── dialog.svelte          # Modal dialogs
├── dropdown-menu.svelte   # Context/dropdown menus
├── tabs.svelte            # Tab navigation
├── progress.svelte        # Progress bar
├── skeleton.svelte        # Loading skeletons
├── separator.svelte       # Horizontal/vertical dividers
├── scroll-area.svelte     # Custom scrollbars
└── index.ts               # Re-exports
```

## Implementation Tasks

- [ ] **Tailwind Configuration**
  - [ ] Create `tailwind.config.js` with custom theme
  - [ ] Configure dark mode as default (`darkMode: 'class'`)
  - [ ] Add custom color palette to `extend.colors`
  - [ ] Configure typography plugin
  - [ ] Set up content paths for Svelte files

- [ ] **CSS Variables Setup**
  - [ ] Create `src/app.css` with CSS custom properties
  - [ ] Define all color tokens as HSL values
  - [ ] Add `.dark` class overrides (for future light mode)
  - [ ] Import Inter and JetBrains Mono fonts

- [ ] **shadcn/ui Initialization**
  - [ ] Run `npx shadcn-svelte@latest init`
  - [ ] Configure bits-ui as component library
  - [ ] Set up `$lib/components/ui` path alias
  - [ ] Generate `components.json` config

- [ ] **Base Component Generation**
  - [ ] Generate Button component with all variants
  - [ ] Generate Card component
  - [ ] Generate Input/Textarea components
  - [ ] Generate Dialog/Modal component
  - [ ] Generate DropdownMenu component
  - [ ] Generate Tabs component
  - [ ] Generate Progress component
  - [ ] Generate Badge component
  - [ ] Generate Tooltip component
  - [ ] Generate ScrollArea component

- [ ] **Icon System**
  - [ ] Install `lucide-svelte`
  - [ ] Create icon size/color utilities
  - [ ] Document common icon usage patterns

- [ ] **Utility Classes**
  - [ ] Create focus ring utility (`.focus-ring`)
  - [ ] Create glass/blur effect utility (`.glass`)
  - [ ] Create gradient utilities for accents
  - [ ] Create animation utilities (fade, slide, scale)

## Acceptance Criteria

- [ ] Running `npm run dev` shows dark-themed UI
- [ ] All shadcn/ui components render with violet accent
- [ ] Tailwind IntelliSense works in VS Code
- [ ] CSS variables can be read via `getComputedStyle()`
- [ ] Components pass accessibility contrast checks (WCAG AA)
- [ ] No CLS (Cumulative Layout Shift) from font loading

## Files to Create/Modify

```
src/
├── app.css                    # Global styles, CSS variables
├── app.html                   # Font imports, dark class
└── lib/
    └── components/
        └── ui/
            ├── button.svelte
            ├── card.svelte
            ├── input.svelte
            ├── textarea.svelte
            ├── select.svelte
            ├── checkbox.svelte
            ├── switch.svelte
            ├── badge.svelte
            ├── avatar.svelte
            ├── tooltip.svelte
            ├── dialog.svelte
            ├── dropdown-menu.svelte
            ├── tabs.svelte
            ├── progress.svelte
            ├── skeleton.svelte
            ├── separator.svelte
            ├── scroll-area.svelte
            └── index.ts

tailwind.config.js             # Tailwind configuration
postcss.config.js              # PostCSS with Tailwind
components.json                # shadcn-svelte config
```

## Testing Checklist

- [ ] Visual regression: Screenshot comparison of components
- [ ] Accessibility: Run axe-core on component showcase
- [ ] Responsiveness: Components scale properly at breakpoints
- [ ] Dark mode: Verify all components respect theme tokens

## Notes

- Use HSL color format for CSS variables (enables opacity modifiers)
- bits-ui is the Svelte-native implementation of Radix primitives
- Consider adding `clsx` or `tailwind-merge` for className composition
- Font files should be self-hosted for offline desktop use
