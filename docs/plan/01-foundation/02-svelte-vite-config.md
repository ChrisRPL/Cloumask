# Svelte 5 and Vite Configuration

> **Parent:** 01-foundation
> **Depends on:** 01-tauri-project-init
> **Blocks:** 03-tailwind-css-setup, 11-frontend-ipc-utils

## Objective

Configure Svelte 5 with SvelteKit and Vite for optimal development experience with hot module replacement (HMR).

## Acceptance Criteria

- [ ] Svelte 5 configured with runes enabled
- [ ] SvelteKit adapter-static for Tauri compatibility
- [ ] Vite dev server runs on port 5173
- [ ] Hot module replacement works for `.svelte` files
- [ ] Basic routing structure in place

## Implementation Steps

1. **Install SvelteKit and dependencies**
   ```bash
   cd /Users/krzysztof/Cloumask
   npm install -D @sveltejs/kit @sveltejs/adapter-static @sveltejs/vite-plugin-svelte svelte vite
   ```

2. **Configure svelte.config.js**
   Create/update `svelte.config.js`:
   ```javascript
   import adapter from '@sveltejs/adapter-static';
   import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

   /** @type {import('@sveltejs/kit').Config} */
   const config = {
     preprocess: vitePreprocess(),
     kit: {
       adapter: adapter({
         pages: 'build',
         assets: 'build',
         fallback: 'index.html',
         precompress: false,
         strict: true
       }),
       alias: {
         $lib: './src/lib',
         $components: './src/lib/components'
       }
     },
     compilerOptions: {
       runes: true
     }
   };

   export default config;
   ```

3. **Configure vite.config.ts**
   Create/update `vite.config.ts`:
   ```typescript
   import { sveltekit } from '@sveltejs/kit/vite';
   import { defineConfig } from 'vite';

   export default defineConfig({
     plugins: [sveltekit()],
     clearScreen: false,
     server: {
       port: 5173,
       strictPort: true,
       watch: {
         ignored: ['**/src-tauri/**']
       }
     },
     envPrefix: ['VITE_', 'TAURI_'],
     build: {
       target: process.env.TAURI_PLATFORM === 'windows'
         ? 'chrome105'
         : 'safari14',
       minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
       sourcemap: !!process.env.TAURI_DEBUG
     }
   });
   ```

4. **Create app.html**
   Create `src/app.html`:
   ```html
   <!doctype html>
   <html lang="en">
     <head>
       <meta charset="utf-8" />
       <link rel="icon" href="%sveltekit.assets%/favicon.png" />
       <meta name="viewport" content="width=device-width, initial-scale=1" />
       <title>Cloumask</title>
       %sveltekit.head%
     </head>
     <body data-sveltekit-preload-data="hover">
       <div style="display: contents">%sveltekit.body%</div>
     </body>
   </html>
   ```

5. **Create root layout**
   Create `src/routes/+layout.svelte`:
   ```svelte
   <script lang="ts">
     import type { Snippet } from 'svelte';

     interface Props {
       children: Snippet;
     }

     let { children }: Props = $props();
   </script>

   {@render children()}
   ```

6. **Create main page**
   Create `src/routes/+page.svelte`:
   ```svelte
   <script lang="ts">
     let greeting = $state('Welcome to Cloumask');
   </script>

   <main>
     <h1>{greeting}</h1>
     <p>Local-first AI for computer vision data processing</p>
   </main>

   <style>
     main {
       display: flex;
       flex-direction: column;
       align-items: center;
       justify-content: center;
       height: 100vh;
       font-family: system-ui, sans-serif;
     }

     h1 {
       font-size: 2rem;
       margin-bottom: 1rem;
     }
   </style>
   ```

7. **Create layout config for static adapter**
   Create `src/routes/+layout.ts`:
   ```typescript
   export const prerender = true;
   export const ssr = false;
   ```

8. **Update package.json scripts**
   Ensure `package.json` has:
   ```json
   {
     "scripts": {
       "dev": "vite dev",
       "build": "vite build",
       "preview": "vite preview",
       "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json"
     }
   }
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `svelte.config.js` | Create | SvelteKit configuration |
| `vite.config.ts` | Create | Vite build configuration |
| `src/app.html` | Create | HTML template |
| `src/routes/+layout.svelte` | Create | Root layout component |
| `src/routes/+layout.ts` | Create | Layout config (SSR off) |
| `src/routes/+page.svelte` | Create | Main page component |
| `package.json` | Modify | Add npm scripts |
| `tsconfig.json` | Create | TypeScript configuration |

## Verification

```bash
# Frontend only
npm run dev
# Open http://localhost:5173 - should show "Welcome to Cloumask"

# Full app
cargo tauri dev
# Window should show the Svelte app

# Test HMR
# Edit +page.svelte greeting text
# Change should appear without full reload
```

## Notes

- Svelte 5 uses runes (`$state`, `$derived`, `$effect`) instead of `$:` reactive statements
- `ssr: false` is required for Tauri (no server-side rendering)
- The `adapter-static` generates a static build suitable for Tauri
- `envPrefix` allows access to `TAURI_*` environment variables in Vite
