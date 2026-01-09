# shadcn/ui Component Library

> **Parent:** 01-foundation
> **Depends on:** 03-tailwind-css-setup
> **Blocks:** 11-frontend-ipc-utils, 12-dev-workflow-ollama

## Objective

Install and configure shadcn-svelte to provide a consistent, accessible component library for the Cloumask UI.

## Acceptance Criteria

- [ ] shadcn-svelte CLI installed and configured
- [ ] `components.json` configured for the project
- [ ] Base components installed (Button, Card, Input)
- [ ] `cn()` utility function available
- [ ] Components render correctly with dark theme

## Implementation Steps

1. **Install shadcn-svelte CLI**
   ```bash
   cd /Users/krzysztof/Cloumask
   npx shadcn-svelte@latest init
   ```

   When prompted:
   - Style: Default
   - Base color: Zinc
   - CSS variables: Yes
   - Global CSS file: `src/app.css`
   - Tailwind config: `tailwind.config.js`
   - Components directory: `src/lib/components/ui`
   - Utils directory: `src/lib/utils`

2. **Verify/update components.json**
   The `components.json` should look like:
   ```json
   {
     "$schema": "https://shadcn-svelte.com/schema.json",
     "style": "default",
     "tailwind": {
       "config": "tailwind.config.js",
       "css": "src/app.css",
       "baseColor": "zinc"
     },
     "aliases": {
       "components": "$lib/components",
       "utils": "$lib/utils"
     }
   }
   ```

3. **Install required dependencies**
   ```bash
   npm install -D bits-ui@next clsx tailwind-merge tailwind-variants
   npm install lucide-svelte
   ```

4. **Create utils.ts with cn() helper**
   Create `src/lib/utils.ts`:
   ```typescript
   import { type ClassValue, clsx } from 'clsx';
   import { twMerge } from 'tailwind-merge';

   export function cn(...inputs: ClassValue[]) {
     return twMerge(clsx(inputs));
   }
   ```

5. **Install base components**
   ```bash
   npx shadcn-svelte@latest add button
   npx shadcn-svelte@latest add card
   npx shadcn-svelte@latest add input
   npx shadcn-svelte@latest add badge
   npx shadcn-svelte@latest add separator
   ```

6. **Create component index file**
   Create `src/lib/components/ui/index.ts`:
   ```typescript
   export { Button, buttonVariants } from './button';
   export { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './card';
   export { Input } from './input';
   export { Badge, badgeVariants } from './badge';
   export { Separator } from './separator';
   ```

7. **Update main page to use components**
   Update `src/routes/+page.svelte`:
   ```svelte
   <script lang="ts">
     import { Button } from '$lib/components/ui/button';
     import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '$lib/components/ui/card';
     import { Input } from '$lib/components/ui/input';
     import { Badge } from '$lib/components/ui/badge';

     let message = $state('');
   </script>

   <main class="flex flex-col items-center justify-center min-h-screen p-8 gap-8">
     <div class="text-center">
       <h1 class="text-4xl font-bold text-text-primary mb-2">Cloumask</h1>
       <p class="text-text-secondary">Local-first AI for computer vision data processing</p>
       <div class="flex gap-2 justify-center mt-4">
         <Badge variant="secondary">Tauri 2.0</Badge>
         <Badge variant="secondary">Svelte 5</Badge>
         <Badge variant="secondary">Python</Badge>
       </div>
     </div>

     <Card class="w-full max-w-md">
       <CardHeader>
         <CardTitle>System Status</CardTitle>
         <CardDescription>Foundation module verification</CardDescription>
       </CardHeader>
       <CardContent class="space-y-4">
         <div class="flex items-center justify-between">
           <span class="text-text-secondary">Frontend</span>
           <Badge variant="default" class="bg-accent-success">Ready</Badge>
         </div>
         <div class="flex items-center justify-between">
           <span class="text-text-secondary">Rust Core</span>
           <Badge variant="outline">Pending</Badge>
         </div>
         <div class="flex items-center justify-between">
           <span class="text-text-secondary">Python Sidecar</span>
           <Badge variant="outline">Pending</Badge>
         </div>
       </CardContent>
     </Card>

     <Card class="w-full max-w-md">
       <CardHeader>
         <CardTitle>Quick Test</CardTitle>
         <CardDescription>Test IPC communication</CardDescription>
       </CardHeader>
       <CardContent class="space-y-4">
         <Input
           placeholder="Type a message..."
           bind:value={message}
         />
         <div class="flex gap-2">
           <Button class="flex-1">Send to Rust</Button>
           <Button variant="secondary" class="flex-1">Check Health</Button>
         </div>
       </CardContent>
     </Card>
   </main>
   ```

8. **Update Tailwind config for shadcn colors**
   Merge shadcn's color requirements into `tailwind.config.js`:
   ```javascript
   /** @type {import('tailwindcss').Config} */
   export default {
     content: ['./src/**/*.{html,js,svelte,ts}'],
     darkMode: 'class',
     theme: {
       extend: {
         colors: {
           // Cloumask design system
           'bg-primary': '#0a0a0b',
           'bg-secondary': '#18181b',
           'bg-tertiary': '#27272a',
           'border': '#3f3f46',
           'text-primary': '#fafafa',
           'text-secondary': '#a1a1aa',
           'text-muted': '#71717a',
           'accent-primary': '#8b5cf6',
           'accent-success': '#22c55e',
           'accent-warning': '#f59e0b',
           'accent-error': '#ef4444',

           // shadcn required colors
           background: '#0a0a0b',
           foreground: '#fafafa',
           card: {
             DEFAULT: '#18181b',
             foreground: '#fafafa'
           },
           popover: {
             DEFAULT: '#18181b',
             foreground: '#fafafa'
           },
           primary: {
             DEFAULT: '#8b5cf6',
             foreground: '#fafafa'
           },
           secondary: {
             DEFAULT: '#27272a',
             foreground: '#fafafa'
           },
           muted: {
             DEFAULT: '#27272a',
             foreground: '#a1a1aa'
           },
           accent: {
             DEFAULT: '#27272a',
             foreground: '#fafafa'
           },
           destructive: {
             DEFAULT: '#ef4444',
             foreground: '#fafafa'
           },
           input: '#3f3f46',
           ring: '#8b5cf6',
         },
         borderRadius: {
           lg: '0.5rem',
           md: 'calc(0.5rem - 2px)',
           sm: 'calc(0.5rem - 4px)'
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

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `components.json` | Create | shadcn-svelte configuration |
| `src/lib/utils.ts` | Create | cn() utility function |
| `src/lib/components/ui/button.svelte` | Create | Button component (via CLI) |
| `src/lib/components/ui/card.svelte` | Create | Card components (via CLI) |
| `src/lib/components/ui/input.svelte` | Create | Input component (via CLI) |
| `src/lib/components/ui/badge.svelte` | Create | Badge component (via CLI) |
| `src/lib/components/ui/index.ts` | Create | Component exports |
| `src/routes/+page.svelte` | Modify | Use shadcn components |
| `tailwind.config.js` | Modify | Add shadcn color aliases |

## Verification

```bash
# Start dev server
npm run dev

# Verify in browser:
# 1. Cards render with correct dark background
# 2. Buttons have proper hover states
# 3. Input focuses with ring color
# 4. Badges show correct variants
# 5. Typography is consistent

# Test component imports
npm run check
# Should pass without import errors
```

## Notes

- shadcn-svelte components are copied into your project (not a dependency)
- Components can be customized directly in `src/lib/components/ui/`
- The `cn()` function merges Tailwind classes safely
- All components are fully accessible (keyboard navigation, ARIA)
- bits-ui provides the headless component primitives
