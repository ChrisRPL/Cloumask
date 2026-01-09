# Development Workflow and Ollama Integration

> **Parent:** 01-foundation
> **Depends on:** All previous specs (01-11)
> **Blocks:** 02-agent-system (Ollama required for LangGraph)

## Objective

Configure the complete development workflow with hot reload and verify Ollama integration for the LLM backend.

## Acceptance Criteria

- [ ] `cargo tauri dev` starts full stack (Tauri + Svelte + Python)
- [ ] Svelte HMR works (changes reflect without full reload)
- [ ] Python uvicorn reload works (backend changes reflect)
- [ ] `package.json` has all necessary scripts
- [ ] Ollama health check endpoint works
- [ ] `ollama list` shows available models from the app
- [ ] All acceptance criteria from SPEC.md are verified

## Implementation Steps

1. **Update package.json scripts**
   Update `package.json`:
   ```json
   {
     "name": "cloumask",
     "version": "0.1.0",
     "private": true,
     "type": "module",
     "scripts": {
       "dev": "vite dev",
       "build": "vite build",
       "preview": "vite preview",
       "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json",
       "check:watch": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json --watch",
       "lint": "eslint .",
       "format": "prettier --write .",
       "tauri": "tauri",
       "tauri:dev": "tauri dev",
       "tauri:build": "tauri build",
       "backend:dev": "cd backend && source .venv/bin/activate && uvicorn api.main:app --reload --port 8765",
       "backend:install": "cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt",
       "test": "vitest",
       "test:e2e": "playwright test"
     },
     "dependencies": {
       "@tauri-apps/api": "^2.0.0"
     },
     "devDependencies": {
       "@sveltejs/adapter-static": "^3.0.0",
       "@sveltejs/kit": "^2.0.0",
       "@sveltejs/vite-plugin-svelte": "^4.0.0",
       "@tauri-apps/cli": "^2.0.0",
       "autoprefixer": "^10.4.0",
       "bits-ui": "^1.0.0",
       "clsx": "^2.1.0",
       "eslint": "^9.0.0",
       "lucide-svelte": "^0.460.0",
       "postcss": "^8.4.0",
       "prettier": "^3.0.0",
       "svelte": "^5.0.0",
       "svelte-check": "^4.0.0",
       "tailwind-merge": "^2.5.0",
       "tailwind-variants": "^0.3.0",
       "tailwindcss": "^3.4.0",
       "typescript": "^5.0.0",
       "vite": "^6.0.0",
       "vitest": "^2.0.0"
     }
   }
   ```

2. **Create Ollama routes in Python**
   Create `backend/api/routes/ollama.py`:
   ```python
   """Ollama integration endpoints for LLM connectivity."""

   import subprocess
   from typing import Literal

   import httpx
   from fastapi import APIRouter, HTTPException
   from pydantic import BaseModel, Field

   from backend.api.config import settings

   router = APIRouter(prefix="/ollama", tags=["Ollama"])


   class OllamaStatus(BaseModel):
       """Ollama service status."""

       available: bool = Field(description="Whether Ollama is reachable")
       url: str = Field(description="Ollama API URL")
       error: str | None = Field(default=None, description="Error message if unavailable")


   class OllamaModel(BaseModel):
       """Information about an Ollama model."""

       name: str = Field(description="Model name (e.g., 'qwen3:14b')")
       size: str = Field(description="Model size on disk")
       modified: str = Field(description="Last modified timestamp")


   class OllamaModelsResponse(BaseModel):
       """Response containing available Ollama models."""

       models: list[OllamaModel] = Field(description="List of available models")
       default_model: str = Field(description="Configured default model")


   class OllamaGenerateRequest(BaseModel):
       """Request to generate text with Ollama."""

       prompt: str = Field(description="The prompt to generate from")
       model: str | None = Field(default=None, description="Model to use (defaults to configured)")
       stream: bool = Field(default=False, description="Whether to stream the response")


   class OllamaGenerateResponse(BaseModel):
       """Response from Ollama generation."""

       response: str = Field(description="Generated text")
       model: str = Field(description="Model used")
       done: bool = Field(description="Whether generation is complete")


   @router.get("/status", response_model=OllamaStatus)
   async def get_ollama_status() -> OllamaStatus:
       """
       Check if Ollama is available and responding.

       Returns the status of the Ollama service.
       """
       try:
           async with httpx.AsyncClient(timeout=5.0) as client:
               response = await client.get(f"{settings.ollama_host}/api/tags")
               if response.status_code == 200:
                   return OllamaStatus(
                       available=True,
                       url=settings.ollama_host,
                   )
               else:
                   return OllamaStatus(
                       available=False,
                       url=settings.ollama_host,
                       error=f"Ollama returned status {response.status_code}",
                   )
       except httpx.ConnectError:
           return OllamaStatus(
               available=False,
               url=settings.ollama_host,
               error="Cannot connect to Ollama. Is it running?",
           )
       except Exception as e:
           return OllamaStatus(
               available=False,
               url=settings.ollama_host,
               error=str(e),
           )


   @router.get("/models", response_model=OllamaModelsResponse)
   async def list_ollama_models() -> OllamaModelsResponse:
       """
       List available Ollama models.

       Equivalent to running `ollama list` on the command line.
       """
       try:
           async with httpx.AsyncClient(timeout=10.0) as client:
               response = await client.get(f"{settings.ollama_host}/api/tags")
               response.raise_for_status()
               data = response.json()

           models = []
           for model in data.get("models", []):
               models.append(
                   OllamaModel(
                       name=model.get("name", "unknown"),
                       size=_format_size(model.get("size", 0)),
                       modified=model.get("modified_at", "unknown"),
                   )
               )

           return OllamaModelsResponse(
               models=models,
               default_model=settings.ollama_model,
           )

       except httpx.ConnectError:
           raise HTTPException(
               status_code=503,
               detail="Cannot connect to Ollama. Is it running?",
           )
       except httpx.HTTPStatusError as e:
           raise HTTPException(
               status_code=e.response.status_code,
               detail=f"Ollama error: {e.response.text}",
           )


   @router.post("/generate", response_model=OllamaGenerateResponse)
   async def generate_text(request: OllamaGenerateRequest) -> OllamaGenerateResponse:
       """
       Generate text using Ollama.

       Simple non-streaming generation for testing.
       """
       model = request.model or settings.ollama_model

       try:
           async with httpx.AsyncClient(timeout=120.0) as client:
               response = await client.post(
                   f"{settings.ollama_host}/api/generate",
                   json={
                       "model": model,
                       "prompt": request.prompt,
                       "stream": False,
                   },
               )
               response.raise_for_status()
               data = response.json()

           return OllamaGenerateResponse(
               response=data.get("response", ""),
               model=model,
               done=data.get("done", True),
           )

       except httpx.ConnectError:
           raise HTTPException(
               status_code=503,
               detail="Cannot connect to Ollama",
           )
       except httpx.HTTPStatusError as e:
           raise HTTPException(
               status_code=e.response.status_code,
               detail=f"Ollama error: {e.response.text}",
           )


   def _format_size(size_bytes: int) -> str:
       """Format bytes to human-readable size."""
       for unit in ["B", "KB", "MB", "GB"]:
           if size_bytes < 1024:
               return f"{size_bytes:.1f} {unit}"
           size_bytes /= 1024
       return f"{size_bytes:.1f} TB"
   ```

3. **Register Ollama routes in main app**
   Update `backend/api/main.py`:
   ```python
   from backend.api.routes import health, ollama

   def create_app() -> FastAPI:
       # ... existing code ...

       # Include routers
       app.include_router(health.router, tags=["Health"])
       app.include_router(ollama.router)

       return app
   ```

4. **Update routes __init__.py**
   Update `backend/api/routes/__init__.py`:
   ```python
   """API route handlers."""

   from backend.api.routes import health, ollama

   __all__ = ["health", "ollama"]
   ```

5. **Add Ollama commands to Rust**
   Create `src-tauri/src/commands/ollama.rs`:
   ```rust
   //! Ollama-related IPC commands.

   use serde::{Deserialize, Serialize};
   use tauri::State;

   use crate::state::AppState;

   #[derive(Debug, Serialize, Deserialize)]
   pub struct OllamaStatus {
       pub available: bool,
       pub url: String,
       pub error: Option<String>,
   }

   #[derive(Debug, Serialize, Deserialize)]
   pub struct OllamaModel {
       pub name: String,
       pub size: String,
       pub modified: String,
   }

   #[derive(Debug, Serialize, Deserialize)]
   pub struct OllamaModelsResponse {
       pub models: Vec<OllamaModel>,
       pub default_model: String,
   }

   /// Check Ollama service status via sidecar.
   #[tauri::command]
   pub async fn get_ollama_status(state: State<'_, AppState>) -> Result<OllamaStatus, String> {
       state
           .sidecar
           .get::<OllamaStatus>("/ollama/status")
           .await
           .map_err(|e| format!("Failed to get Ollama status: {}", e))
   }

   /// List available Ollama models via sidecar.
   #[tauri::command]
   pub async fn list_ollama_models(
       state: State<'_, AppState>,
   ) -> Result<OllamaModelsResponse, String> {
       state
           .sidecar
           .get::<OllamaModelsResponse>("/ollama/models")
           .await
           .map_err(|e| format!("Failed to list Ollama models: {}", e))
   }
   ```

6. **Register Ollama commands**
   Update `src-tauri/src/commands/mod.rs`:
   ```rust
   mod ollama;
   mod sidecar;
   mod system;

   pub use ollama::*;
   pub use sidecar::*;
   pub use system::*;
   ```

   Update `src-tauri/src/lib.rs`:
   ```rust
   use commands::{
       call_sidecar_get, check_health, echo, get_app_info, get_ollama_status,
       get_sidecar_status, list_ollama_models, ping, restart_sidecar,
       start_sidecar, stop_sidecar,
   };

   // In invoke_handler:
   .invoke_handler(tauri::generate_handler![
       // System commands
       ping,
       echo,
       get_app_info,
       // Sidecar commands
       get_sidecar_status,
       start_sidecar,
       stop_sidecar,
       restart_sidecar,
       check_health,
       call_sidecar_get,
       // Ollama commands
       get_ollama_status,
       list_ollama_models,
   ])
   ```

7. **Add Ollama utilities to frontend**
   Update `src/lib/utils/tauri.ts`:
   ```typescript
   // Add to imports
   import type { OllamaStatus, OllamaModel } from '$lib/types/ipc';

   // Add Ollama functions
   export async function getOllamaStatus(): Promise<OllamaStatus> {
     return invokeCommand<OllamaStatus>('get_ollama_status');
   }

   export async function listOllamaModels(): Promise<OllamaModelsResponse> {
     return invokeCommand<OllamaModelsResponse>('list_ollama_models');
   }
   ```

   Update `src/lib/types/ipc.ts`:
   ```typescript
   // Add Ollama types
   export interface OllamaStatus {
     available: boolean;
     url: string;
     error: string | null;
   }

   export interface OllamaModel {
     name: string;
     size: string;
     modified: string;
   }

   export interface OllamaModelsResponse {
     models: OllamaModel[];
     default_model: string;
   }
   ```

8. **Update main page to show Ollama status**
   Add Ollama section to `src/routes/+page.svelte`:
   ```svelte
   <!-- Add to script -->
   import { getOllamaStatus, listOllamaModels } from '$lib/utils/tauri';
   import type { OllamaStatus, OllamaModelsResponse } from '$lib/types/ipc';

   let ollamaStatus = $state<OllamaStatus | null>(null);
   let ollamaModels = $state<OllamaModelsResponse | null>(null);

   // Add to refreshStatus function
   const [/* existing */, ollamaResult, modelsResult] = await Promise.allSettled([
     /* existing calls */,
     getOllamaStatus(),
     listOllamaModels(),
   ]);

   if (ollamaResult.status === 'fulfilled') {
     ollamaStatus = ollamaResult.value;
   }
   if (modelsResult.status === 'fulfilled') {
     ollamaModels = modelsResult.value;
   }

   <!-- Add to System Status card -->
   <div class="flex items-center justify-between">
     <span class="text-text-secondary">Ollama LLM</span>
     <Badge
       variant={getBadgeVariant(ollamaStatus?.available ? 'healthy' : 'unhealthy')}
       class={ollamaStatus?.available ? 'bg-accent-success' : ''}
     >
       {ollamaStatus?.available ? 'Connected' : 'Unavailable'}
     </Badge>
   </div>

   <!-- Add Ollama Models card -->
   {#if ollamaModels && ollamaModels.models.length > 0}
     <Card class="w-full max-w-md">
       <CardHeader>
         <CardTitle>Ollama Models</CardTitle>
         <CardDescription>Default: {ollamaModels.default_model}</CardDescription>
       </CardHeader>
       <CardContent>
         <div class="space-y-2">
           {#each ollamaModels.models as model}
             <div class="flex justify-between text-sm">
               <span class="text-text-primary font-mono">{model.name}</span>
               <span class="text-text-muted">{model.size}</span>
             </div>
           {/each}
         </div>
       </CardContent>
     </Card>
   {/if}
   ```

9. **Create verification checklist script**
   Create `scripts/verify-foundation.sh`:
   ```bash
   #!/bin/bash
   # Verification script for Foundation module

   set -e

   echo "=== Foundation Module Verification ==="
   echo ""

   # Colors
   GREEN='\033[0;32m'
   RED='\033[0;31m'
   NC='\033[0m'

   check() {
     if [ $? -eq 0 ]; then
       echo -e "${GREEN}✓${NC} $1"
     else
       echo -e "${RED}✗${NC} $1"
       exit 1
     fi
   }

   echo "1. Checking Node.js dependencies..."
   npm list @tauri-apps/api > /dev/null 2>&1
   check "Node dependencies installed"

   echo ""
   echo "2. Checking Rust compilation..."
   cd src-tauri && cargo check > /dev/null 2>&1
   check "Rust code compiles"
   cd ..

   echo ""
   echo "3. Checking Python backend..."
   cd backend
   source .venv/bin/activate 2>/dev/null || python3 -m venv .venv && source .venv/bin/activate
   python -c "from backend.api.main import app" 2>/dev/null
   check "Python backend imports"
   cd ..

   echo ""
   echo "4. Checking Svelte compilation..."
   npm run check > /dev/null 2>&1
   check "Svelte type check passes"

   echo ""
   echo "5. Starting backend for health check..."
   cd backend
   uvicorn api.main:app --port 8765 &
   BACKEND_PID=$!
   sleep 3

   curl -s http://localhost:8765/health | grep -q "healthy"
   check "Backend health endpoint works"

   curl -s http://localhost:8765/ready | grep -q "ready"
   check "Backend ready endpoint works"

   kill $BACKEND_PID 2>/dev/null
   cd ..

   echo ""
   echo "6. Checking Ollama connectivity..."
   curl -s http://localhost:11434/api/tags > /dev/null 2>&1
   if [ $? -eq 0 ]; then
     echo -e "${GREEN}✓${NC} Ollama is running"
   else
     echo -e "${RED}!${NC} Ollama not running (optional - start with 'ollama serve')"
   fi

   echo ""
   echo "=== All Foundation checks passed! ==="
   echo ""
   echo "Next steps:"
   echo "  1. Run 'cargo tauri dev' to start the full application"
   echo "  2. Verify the UI shows all components as 'healthy'"
   echo "  3. Test IPC commands in browser devtools"
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `package.json` | Modify | Update scripts |
| `backend/api/routes/ollama.py` | Create | Ollama API endpoints |
| `backend/api/main.py` | Modify | Register Ollama routes |
| `backend/api/routes/__init__.py` | Modify | Export Ollama module |
| `src-tauri/src/commands/ollama.rs` | Create | Ollama IPC commands |
| `src-tauri/src/commands/mod.rs` | Modify | Export Ollama commands |
| `src-tauri/src/lib.rs` | Modify | Register Ollama commands |
| `src/lib/utils/tauri.ts` | Modify | Add Ollama functions |
| `src/lib/types/ipc.ts` | Modify | Add Ollama types |
| `src/routes/+page.svelte` | Modify | Show Ollama status |
| `scripts/verify-foundation.sh` | Create | Verification script |

## Verification

### Manual Verification

```bash
# 1. Start Ollama (if not running)
ollama serve

# 2. Pull a test model (if needed)
ollama pull qwen3:14b  # Or smaller: ollama pull llama3.1:8b

# 3. Start the full application
cargo tauri dev

# 4. In the app window, verify:
#    - Frontend shows "healthy" (green badge)
#    - Rust Core shows "healthy"
#    - Python Sidecar shows "healthy"
#    - Ollama LLM shows "Connected"
#    - Ollama Models card shows available models

# 5. Test HMR - edit src/routes/+page.svelte:
#    - Change the title text
#    - Save the file
#    - Change should appear without page reload

# 6. Test Python reload:
#    - Edit backend/api/routes/health.py
#    - Change the version string
#    - Save and wait a moment
#    - Click Refresh in the app
#    - New version should appear
```

### Automated Verification

```bash
chmod +x scripts/verify-foundation.sh
./scripts/verify-foundation.sh
```

### Acceptance Criteria Checklist

From the original SPEC.md:

- [x] `cargo tauri dev` starts the full application
- [x] Frontend can call Rust commands via IPC
- [x] Rust can communicate with Python sidecar
- [x] Sidecar starts automatically and stops on app close
- [x] Hot reload works for Svelte and Python changes
- [x] `ollama list` shows available models from the app

## Notes

- Ollama must be installed and running separately (`ollama serve`)
- The default model is configurable via `CLOUMASK_OLLAMA_MODEL` env var
- HTTP timeouts are generous (120s) for large model responses
- The verification script can be run as a CI check
- Consider adding `npm run verify` script that runs all checks
- The foundation is now complete and ready for the Agent System module
