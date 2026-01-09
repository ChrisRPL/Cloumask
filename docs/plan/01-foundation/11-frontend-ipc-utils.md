# Frontend IPC Utilities

> **Parent:** 01-foundation
> **Depends on:** 04-shadcn-ui-components, 09-tauri-ipc-commands, 10-rust-python-http
> **Blocks:** 12-dev-workflow-ollama

## Objective

Create TypeScript utilities for type-safe communication with Tauri IPC commands, providing a clean API for frontend components.

## Acceptance Criteria

- [ ] `src/lib/utils/tauri.ts` with typed invoke wrapper
- [ ] `src/lib/types/ipc.ts` with all IPC types
- [ ] `checkHealth()`, `getSidecarStatus()` functions work
- [ ] Error handling returns user-friendly messages
- [ ] Main page shows live status of all system components

## Implementation Steps

1. **Create IPC types module**
   Create `src/lib/types/ipc.ts`:
   ```typescript
   /**
    * Type definitions for Tauri IPC communication.
    * These types mirror the Rust structs returned by commands.
    */

   /** Health status of a component */
   export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'not_loaded';

   /** Response from the sidecar health endpoint */
   export interface HealthResponse {
     status: HealthStatus;
     version: string;
     timestamp: string;
     components: Record<string, string>;
   }

   /** Status of the Python sidecar process */
   export interface SidecarStatus {
     running: boolean;
     url: string;
     port: number;
   }

   /** Application information */
   export interface AppInfo {
     name: string;
     version: string;
     tauri_version: string;
     debug: boolean;
   }

   /** Combined system status for UI display */
   export interface SystemStatus {
     frontend: HealthStatus;
     rust: HealthStatus;
     python: HealthStatus;
     ollama: HealthStatus;
   }

   /** IPC error with context */
   export interface IPCError {
     command: string;
     message: string;
     details?: string;
   }
   ```

2. **Create Tauri utilities module**
   Create `src/lib/utils/tauri.ts`:
   ```typescript
   /**
    * Tauri IPC utilities with type-safe command invocation.
    */

   import { invoke } from '@tauri-apps/api/core';
   import type {
     AppInfo,
     HealthResponse,
     IPCError,
     SidecarStatus,
   } from '$lib/types/ipc';

   /**
    * Type-safe wrapper around Tauri invoke.
    * Converts errors to user-friendly IPCError objects.
    */
   async function invokeCommand<T>(
     command: string,
     args?: Record<string, unknown>
   ): Promise<T> {
     try {
       return await invoke<T>(command, args);
     } catch (error) {
       const message = error instanceof Error ? error.message : String(error);
       throw {
         command,
         message: `Command '${command}' failed`,
         details: message,
       } as IPCError;
     }
   }

   // ============================================
   // System Commands
   // ============================================

   /** Test IPC connectivity with a simple ping. */
   export async function ping(): Promise<string> {
     return invokeCommand<string>('ping');
   }

   /** Echo a message back (for testing). */
   export async function echo(message: string): Promise<string> {
     return invokeCommand<string>('echo', { message });
   }

   /** Get application information. */
   export async function getAppInfo(): Promise<AppInfo> {
     return invokeCommand<AppInfo>('get_app_info');
   }

   // ============================================
   // Sidecar Commands
   // ============================================

   /** Get the current status of the Python sidecar. */
   export async function getSidecarStatus(): Promise<SidecarStatus> {
     return invokeCommand<SidecarStatus>('get_sidecar_status');
   }

   /** Start the Python sidecar if not running. */
   export async function startSidecar(): Promise<string> {
     return invokeCommand<string>('start_sidecar');
   }

   /** Stop the Python sidecar. */
   export async function stopSidecar(): Promise<string> {
     return invokeCommand<string>('stop_sidecar');
   }

   /** Restart the Python sidecar. */
   export async function restartSidecar(): Promise<string> {
     return invokeCommand<string>('restart_sidecar');
   }

   /** Check the health of the Python sidecar. */
   export async function checkHealth(): Promise<HealthResponse> {
     return invokeCommand<HealthResponse>('check_health');
   }

   /** Call a generic sidecar GET endpoint. */
   export async function callSidecarGet<T = unknown>(endpoint: string): Promise<T> {
     return invokeCommand<T>('call_sidecar_get', { endpoint });
   }

   // ============================================
   // Utility Functions
   // ============================================

   /**
    * Check if we're running inside Tauri.
    * Useful for SSR compatibility and testing.
    */
   export function isTauri(): boolean {
     return typeof window !== 'undefined' && '__TAURI__' in window;
   }

   /**
    * Wait for a condition with timeout.
    */
   export async function waitFor(
     condition: () => Promise<boolean>,
     options: { timeout?: number; interval?: number } = {}
   ): Promise<boolean> {
     const { timeout = 10000, interval = 500 } = options;
     const start = Date.now();

     while (Date.now() - start < timeout) {
       if (await condition()) {
         return true;
       }
       await new Promise((resolve) => setTimeout(resolve, interval));
     }

     return false;
   }

   /**
    * Wait for the sidecar to become healthy.
    */
   export async function waitForSidecar(timeout = 10000): Promise<boolean> {
     return waitFor(async () => {
       try {
         const health = await checkHealth();
         return health.status === 'healthy';
       } catch {
         return false;
       }
     }, { timeout });
   }
   ```

3. **Create types index file**
   Create `src/lib/types/index.ts`:
   ```typescript
   export * from './ipc';
   ```

4. **Create utils index file**
   Create `src/lib/utils/index.ts`:
   ```typescript
   export * from './tauri';
   ```

5. **Update main page to show system status**
   Update `src/routes/+page.svelte`:
   ```svelte
   <script lang="ts">
     import { onMount } from 'svelte';
     import { Button } from '$lib/components/ui/button';
     import {
       Card,
       CardContent,
       CardHeader,
       CardTitle,
       CardDescription,
     } from '$lib/components/ui/card';
     import { Badge } from '$lib/components/ui/badge';
     import { Separator } from '$lib/components/ui/separator';
     import {
       getAppInfo,
       getSidecarStatus,
       checkHealth,
       restartSidecar,
       isTauri,
     } from '$lib/utils/tauri';
     import type { AppInfo, HealthResponse, SidecarStatus } from '$lib/types/ipc';

     // State
     let appInfo = $state<AppInfo | null>(null);
     let sidecarStatus = $state<SidecarStatus | null>(null);
     let healthResponse = $state<HealthResponse | null>(null);
     let error = $state<string | null>(null);
     let loading = $state(false);

     // Derived states
     let frontendStatus = $derived<'healthy' | 'unhealthy'>('healthy');
     let rustStatus = $derived<'healthy' | 'unhealthy'>(appInfo ? 'healthy' : 'unhealthy');
     let pythonStatus = $derived<'healthy' | 'unhealthy' | 'loading'>(
       loading ? 'loading' : healthResponse?.status === 'healthy' ? 'healthy' : 'unhealthy'
     );

     // Load status on mount
     onMount(() => {
       if (isTauri()) {
         refreshStatus();
       }
     });

     async function refreshStatus() {
       loading = true;
       error = null;

       try {
         // Fetch all status in parallel
         const [appResult, sidecarResult, healthResult] = await Promise.allSettled([
           getAppInfo(),
           getSidecarStatus(),
           checkHealth(),
         ]);

         if (appResult.status === 'fulfilled') {
           appInfo = appResult.value;
         }

         if (sidecarResult.status === 'fulfilled') {
           sidecarStatus = sidecarResult.value;
         }

         if (healthResult.status === 'fulfilled') {
           healthResponse = healthResult.value;
         } else {
           error = 'Sidecar health check failed. Is Python running?';
         }
       } catch (e) {
         error = e instanceof Error ? e.message : 'Unknown error';
       } finally {
         loading = false;
       }
     }

     async function handleRestartSidecar() {
       loading = true;
       error = null;

       try {
         await restartSidecar();
         // Wait a bit for restart, then refresh
         await new Promise((resolve) => setTimeout(resolve, 2000));
         await refreshStatus();
       } catch (e) {
         error = e instanceof Error ? e.message : 'Failed to restart sidecar';
         loading = false;
       }
     }

     function getBadgeVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
       switch (status) {
         case 'healthy':
           return 'default';
         case 'loading':
           return 'secondary';
         case 'unhealthy':
           return 'destructive';
         default:
           return 'outline';
       }
     }
   </script>

   <main class="flex flex-col items-center justify-center min-h-screen p-8 gap-8">
     <!-- Header -->
     <div class="text-center">
       <h1 class="text-4xl font-bold text-text-primary mb-2">Cloumask</h1>
       <p class="text-text-secondary">Local-first AI for computer vision data processing</p>
       <div class="flex gap-2 justify-center mt-4">
         <Badge variant="secondary">Tauri 2.0</Badge>
         <Badge variant="secondary">Svelte 5</Badge>
         <Badge variant="secondary">Python</Badge>
       </div>
     </div>

     <!-- System Status Card -->
     <Card class="w-full max-w-md">
       <CardHeader>
         <CardTitle class="flex items-center justify-between">
           System Status
           <Button variant="ghost" size="sm" onclick={refreshStatus} disabled={loading}>
             {loading ? 'Checking...' : 'Refresh'}
           </Button>
         </CardTitle>
         <CardDescription>Foundation module verification</CardDescription>
       </CardHeader>
       <CardContent class="space-y-4">
         <!-- Frontend Status -->
         <div class="flex items-center justify-between">
           <span class="text-text-secondary">Frontend (Svelte 5)</span>
           <Badge variant={getBadgeVariant(frontendStatus)} class={frontendStatus === 'healthy' ? 'bg-accent-success' : ''}>
             {frontendStatus}
           </Badge>
         </div>

         <!-- Rust Status -->
         <div class="flex items-center justify-between">
           <span class="text-text-secondary">Rust Core (Tauri)</span>
           <Badge variant={getBadgeVariant(rustStatus)} class={rustStatus === 'healthy' ? 'bg-accent-success' : ''}>
             {rustStatus}
           </Badge>
         </div>

         <!-- Python Status -->
         <div class="flex items-center justify-between">
           <span class="text-text-secondary">Python Sidecar</span>
           <Badge variant={getBadgeVariant(pythonStatus)} class={pythonStatus === 'healthy' ? 'bg-accent-success' : ''}>
             {pythonStatus}
           </Badge>
         </div>

         <Separator />

         <!-- Sidecar Details -->
         {#if sidecarStatus}
           <div class="text-sm text-text-muted space-y-1">
             <p>Process: {sidecarStatus.running ? 'Running' : 'Stopped'}</p>
             <p>URL: {sidecarStatus.url}</p>
             <p>Port: {sidecarStatus.port}</p>
           </div>
         {/if}

         <!-- Health Details -->
         {#if healthResponse}
           <div class="text-sm text-text-muted space-y-1">
             <p>Version: {healthResponse.version}</p>
             <p>Last check: {new Date(healthResponse.timestamp).toLocaleTimeString()}</p>
           </div>
         {/if}

         <!-- Error Display -->
         {#if error}
           <div class="p-3 rounded-md bg-accent-error/10 border border-accent-error/20">
             <p class="text-sm text-accent-error">{error}</p>
           </div>
         {/if}
       </CardContent>
     </Card>

     <!-- App Info Card -->
     {#if appInfo}
       <Card class="w-full max-w-md">
         <CardHeader>
           <CardTitle>Application Info</CardTitle>
         </CardHeader>
         <CardContent class="space-y-2 text-sm text-text-secondary">
           <p><span class="text-text-primary">Name:</span> {appInfo.name}</p>
           <p><span class="text-text-primary">Version:</span> {appInfo.version}</p>
           <p><span class="text-text-primary">Tauri:</span> {appInfo.tauri_version}</p>
           <p><span class="text-text-primary">Mode:</span> {appInfo.debug ? 'Development' : 'Production'}</p>
         </CardContent>
       </Card>
     {/if}

     <!-- Actions -->
     <div class="flex gap-4">
       <Button onclick={handleRestartSidecar} disabled={loading}>
         Restart Sidecar
       </Button>
       <Button variant="secondary" onclick={() => window.open('http://localhost:8765/docs', '_blank')}>
         API Docs
       </Button>
     </div>
   </main>
   ```

6. **Install Tauri API package**
   ```bash
   npm install @tauri-apps/api
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/lib/types/ipc.ts` | Create | IPC type definitions |
| `src/lib/types/index.ts` | Create | Types barrel export |
| `src/lib/utils/tauri.ts` | Create | Tauri IPC utilities |
| `src/lib/utils/index.ts` | Create | Utils barrel export |
| `src/routes/+page.svelte` | Modify | System status dashboard |
| `package.json` | Modify | Add @tauri-apps/api |

## Verification

```bash
# Start the full application
cargo tauri dev

# In the app window:
# 1. Page shows "Cloumask" header
# 2. System Status card shows Frontend, Rust, Python statuses
# 3. Click "Refresh" - statuses update
# 4. All three should show "healthy" (green badges)
# 5. Click "Restart Sidecar" - sidecar restarts, status updates
# 6. Click "API Docs" - opens FastAPI Swagger UI

# In browser devtools console:
import { ping, checkHealth } from '$lib/utils/tauri';
await ping()  // "pong"
await checkHealth()  // { status: "healthy", ... }
```

## Notes

- The `@tauri-apps/api` package provides TypeScript bindings for Tauri
- `Promise.allSettled` allows parallel fetching without failing on first error
- The `isTauri()` check enables SSR compatibility and testing without Tauri
- Svelte 5 runes (`$state`, `$derived`) are used for reactivity
- Error states are displayed with appropriate styling
- Consider adding periodic auto-refresh for continuous monitoring
