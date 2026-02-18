# Tailwind CSS Integration

> **Parent:** 01-foundation
> **Depends on:** 02-svelte-vite-config
> **Blocks:** 04-shadcn-ui-components

## Objective

Integrate Tailwind CSS with the Cloumask design system colors for consistent styling across the application.

## Acceptance Criteria

- [ ] Tailwind CSS installed and configured
- [ ] PostCSS processing works with Vite
- [ ] Design system colors available as Tailwind utilities
- [ ] Light theme is the default
- [ ] Global styles applied correctly

## Implementation Steps

1. **Install Tailwind CSS and dependencies**
   ```bash
   cd /Users/krzysztof/Cloumask
   npm install -D tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```

2. **Configure tailwind.config.js**
   Update `tailwind.config.js`:
   ```javascript
   /** @type {import('tailwindcss').Config} */
   export default {
     content: ['./src/**/*.{html,js,svelte,ts}'],
     darkMode: 'class',
     theme: {
       extend: {
         colors: {
           // Background colors
           'bg-primary': '#0a0a0b',
           'bg-secondary': '#18181b',
           'bg-tertiary': '#27272a',

           // Border colors
           'border': '#3f3f46',

           // Text colors
           'text-primary': '#fafafa',
           'text-secondary': '#a1a1aa',
           'text-muted': '#71717a',

           // Accent colors
           'accent-primary': '#8b5cf6',
           'accent-success': '#22c55e',
           'accent-warning': '#f59e0b',
           'accent-error': '#ef4444',
         },
         fontFamily: {
           sans: ['Inter', 'system-ui', 'sans-serif'],
           mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
         },
       },
     },
     plugins: [],
   };
   ```

3. **Configure postcss.config.js**
   Update `postcss.config.js`:
   ```javascript
   export default {
     plugins: {
       tailwindcss: {},
       autoprefixer: {},
     },
   };
   ```

4. **Create global CSS file**
   Create `src/app.css`:
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;

   /* Custom base styles */
   @layer base {
     :root {
       --bg-primary: #0a0a0b;
       --bg-secondary: #18181b;
       --bg-tertiary: #27272a;
       --border: #3f3f46;
       --text-primary: #fafafa;
       --text-secondary: #a1a1aa;
       --text-muted: #71717a;
       --accent-primary: #8b5cf6;
       --accent-success: #22c55e;
       --accent-warning: #f59e0b;
       --accent-error: #ef4444;
     }

     html {
       background-color: var(--bg-primary);
       color: var(--text-primary);
     }

     body {
       @apply bg-bg-primary text-text-primary antialiased;
       font-family: 'Inter', system-ui, sans-serif;
     }

     /* Scrollbar styling for dark theme */
     ::-webkit-scrollbar {
       width: 8px;
       height: 8px;
     }

     ::-webkit-scrollbar-track {
       background: var(--bg-secondary);
     }

     ::-webkit-scrollbar-thumb {
       background: var(--border);
       border-radius: 4px;
     }

     ::-webkit-scrollbar-thumb:hover {
       background: var(--text-muted);
     }
   }

   /* Component utilities */
   @layer components {
     .card {
       @apply bg-bg-secondary border border-border rounded-lg;
     }

     .btn-primary {
       @apply bg-accent-primary text-white px-4 py-2 rounded-md
              hover:opacity-90 transition-opacity;
     }

     .input-base {
       @apply bg-bg-tertiary border border-border rounded-md px-3 py-2
              text-text-primary placeholder:text-text-muted
              focus:outline-none focus:ring-2 focus:ring-accent-primary;
     }
   }
   ```

5. **Import CSS in layout**
   Update `src/routes/+layout.svelte`:
   ```svelte
   <script lang="ts">
     import '../app.css';
     import type { Snippet } from 'svelte';

     interface Props {
       children: Snippet;
     }

     let { children }: Props = $props();
   </script>

   <div class="min-h-screen bg-bg-primary">
     {@render children()}
   </div>
   ```

6. **Update main page with Tailwind classes**
   Update `src/routes/+page.svelte`:
   ```svelte
   <script lang="ts">
     let greeting = $state('Welcome to Cloumask');
   </script>

   <main class="flex flex-col items-center justify-center h-screen">
     <h1 class="text-4xl font-bold text-text-primary mb-4">{greeting}</h1>
     <p class="text-text-secondary">Local-first AI for computer vision data processing</p>

     <div class="mt-8 flex gap-4">
       <button class="btn-primary">Get Started</button>
       <button class="card px-4 py-2 hover:bg-bg-tertiary transition-colors">
         Learn More
       </button>
     </div>
   </main>
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `tailwind.config.js` | Create | Tailwind configuration with design system |
| `postcss.config.js` | Create | PostCSS plugins |
| `src/app.css` | Create | Global styles and Tailwind directives |
| `src/routes/+layout.svelte` | Modify | Import global CSS |
| `src/routes/+page.svelte` | Modify | Use Tailwind classes |

## Verification

```bash
# Start dev server
npm run dev

# Check that:
# 1. Background is dark (#0a0a0b)
# 2. Text is light (#fafafa)
# 3. Buttons have correct styling
# 4. Hover states work

# Verify Tailwind compilation
# Open browser DevTools > Elements
# Check that Tailwind classes generate correct CSS
```

## Notes

- The design system uses a light theme by default with optional dark mode
- CSS variables are defined for JavaScript access if needed
- Custom component classes (`card`, `btn-primary`, `input-base`) provide reusable patterns
- The `darkMode: 'class'` setting allows future light mode toggle support
