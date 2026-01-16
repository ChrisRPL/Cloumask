"""FastAPI application entry point for Cloumask sidecar."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.api.routes import health, ollama, review, scripts
from backend.api.streaming import endpoints as streaming


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
            "tauri://localhost",  # Tauri production
            "https://tauri.localhost",  # Tauri production (alternative)
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(ollama.router)
    app.include_router(scripts.router)
    app.include_router(review.router)
    app.include_router(streaming.router)

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
