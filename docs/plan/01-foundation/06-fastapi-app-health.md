# FastAPI Application with Health Endpoint

> **Parent:** 01-foundation
> **Depends on:** 05-python-backend-structure
> **Blocks:** 07-python-dependencies, 10-rust-python-http

## Objective

Create the FastAPI application instance with CORS configuration and a health check endpoint that the Rust core can use to verify sidecar status.

## Acceptance Criteria

- [ ] FastAPI app created in `backend/api/main.py`
- [ ] CORS middleware configured for localhost origins
- [ ] `/health` endpoint returns structured JSON response
- [ ] Response includes status, version, and timestamp
- [ ] `uvicorn api.main:app --port 8765` starts successfully

## Implementation Steps

1. **Create the FastAPI application**
   Create `backend/api/main.py`:
   ```python
   """FastAPI application entry point for Cloumask sidecar."""

   from contextlib import asynccontextmanager
   from typing import AsyncGenerator

   from fastapi import FastAPI
   from fastapi.middleware.cors import CORSMiddleware

   from backend import __version__
   from backend.api.routes import health


   @asynccontextmanager
   async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
       """Application lifespan handler for startup/shutdown."""
       # Startup
       print(f"Cloumask Backend v{__version__} starting...")
       yield
       # Shutdown
       print("Cloumask Backend shutting down...")


   def create_app() -> FastAPI:
       """Create and configure the FastAPI application."""
       app = FastAPI(
           title="Cloumask Backend",
           description="Python sidecar for CV processing and LangGraph agent",
           version=__version__,
           lifespan=lifespan,
           docs_url="/docs",
           redoc_url="/redoc",
       )

       # Configure CORS for Tauri frontend
       app.add_middleware(
           CORSMiddleware,
           allow_origins=[
               "http://localhost:5173",  # Vite dev server
               "http://localhost:1420",  # Tauri dev server
               "tauri://localhost",      # Tauri production
               "https://tauri.localhost", # Tauri production (alternative)
           ],
           allow_credentials=True,
           allow_methods=["*"],
           allow_headers=["*"],
       )

       # Include routers
       app.include_router(health.router, tags=["Health"])

       return app


   # Create the application instance
   app = create_app()


   def start() -> None:
       """Entry point for the cloumask-backend CLI command."""
       import uvicorn

       uvicorn.run(
           "backend.api.main:app",
           host="127.0.0.1",
           port=8765,
           reload=False,
           log_level="info",
       )


   if __name__ == "__main__":
       import uvicorn

       uvicorn.run(
           "backend.api.main:app",
           host="127.0.0.1",
           port=8765,
           reload=True,
           log_level="debug",
       )
   ```

2. **Create the health endpoint**
   Create `backend/api/routes/health.py`:
   ```python
   """Health check endpoint for sidecar status verification."""

   from datetime import datetime, timezone
   from typing import Literal

   from fastapi import APIRouter
   from pydantic import BaseModel, Field

   from backend import __version__


   router = APIRouter()


   class HealthResponse(BaseModel):
       """Response model for health check endpoint."""

       status: Literal["healthy", "degraded", "unhealthy"] = Field(
           description="Current health status of the sidecar"
       )
       version: str = Field(description="Backend version")
       timestamp: str = Field(description="ISO 8601 timestamp")
       components: dict[str, str] = Field(
           default_factory=dict,
           description="Status of individual components"
       )


   class ReadyResponse(BaseModel):
       """Response model for readiness check."""

       ready: bool = Field(description="Whether the sidecar is ready to accept requests")
       checks: dict[str, bool] = Field(
           default_factory=dict,
           description="Individual readiness checks"
       )


   @router.get("/health", response_model=HealthResponse)
   async def health_check() -> HealthResponse:
       """
       Check the health status of the sidecar.

       Returns the current status, version, and component health.
       Used by the Rust core to verify sidecar is running.
       """
       return HealthResponse(
           status="healthy",
           version=__version__,
           timestamp=datetime.now(timezone.utc).isoformat(),
           components={
               "api": "healthy",
               "agent": "not_loaded",  # Will be updated in 02-agent-system
               "cv_models": "not_loaded",  # Will be updated in 03-cv-models
           },
       )


   @router.get("/ready", response_model=ReadyResponse)
   async def readiness_check() -> ReadyResponse:
       """
       Check if the sidecar is ready to process requests.

       Used for startup probes to ensure sidecar is fully initialized.
       """
       # Basic checks for foundation module
       checks = {
           "api_running": True,
           "routes_loaded": True,
       }

       return ReadyResponse(
           ready=all(checks.values()),
           checks=checks,
       )


   @router.get("/")
   async def root() -> dict[str, str]:
       """Root endpoint with API information."""
       return {
           "name": "Cloumask Backend",
           "version": __version__,
           "docs": "/docs",
       }
   ```

3. **Update routes __init__.py**
   Update `backend/api/routes/__init__.py`:
   ```python
   """API route handlers."""

   from backend.api.routes import health

   __all__ = ["health"]
   ```

4. **Create a config module for settings**
   Create `backend/api/config.py`:
   ```python
   """Application configuration using pydantic-settings."""

   from pydantic_settings import BaseSettings, SettingsConfigDict


   class Settings(BaseSettings):
       """Application settings loaded from environment variables."""

       model_config = SettingsConfigDict(
           env_prefix="CLOUMASK_",
           env_file=".env",
           env_file_encoding="utf-8",
       )

       # Server settings
       host: str = "127.0.0.1"
       port: int = 8765
       debug: bool = False

       # Ollama settings (for 12-dev-workflow-ollama)
       ollama_host: str = "http://localhost:11434"
       ollama_model: str = "qwen3:14b"

       # Model settings (for 03-cv-models)
       models_dir: str = "models"
       device: str = "cuda"  # or "cpu", "mps"


   # Global settings instance
   settings = Settings()
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/api/main.py` | Create | FastAPI application factory |
| `backend/api/routes/health.py` | Create | Health check endpoints |
| `backend/api/routes/__init__.py` | Modify | Export health module |
| `backend/api/config.py` | Create | Application settings |

## Verification

```bash
cd /Users/krzysztof/Cloumask/backend

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not done)
pip install fastapi uvicorn pydantic pydantic-settings

# Start the server
uvicorn api.main:app --reload --port 8765

# In another terminal, test the endpoints:
curl http://localhost:8765/
# {"name":"Cloumask Backend","version":"0.1.0","docs":"/docs"}

curl http://localhost:8765/health
# {"status":"healthy","version":"0.1.0","timestamp":"...","components":{...}}

curl http://localhost:8765/ready
# {"ready":true,"checks":{"api_running":true,"routes_loaded":true}}

# Open http://localhost:8765/docs for Swagger UI
```

## Notes

- CORS is configured for both development (localhost:5173/1420) and production (tauri://)
- The health endpoint returns structured data for programmatic checks
- The readiness endpoint is separate from health for Kubernetes-style probes
- Settings are loaded from environment variables with `CLOUMASK_` prefix
- The lifespan context manager handles startup/shutdown logic
- FastAPI's automatic OpenAPI docs are available at `/docs`
