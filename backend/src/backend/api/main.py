"""FastAPI application entry point for Cloumask sidecar."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.agent.llm.config import REQUIRED_MODEL
from backend.api.routes import detect3d, fusion, health, llm, pointcloud, review, rosbag, scripts
from backend.api.routes.llm import check_llm_ready_on_startup
from backend.api.streaming import endpoints as streaming

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print(f"Cloumask Backend v{__version__} starting...")

    # Check LLM service availability
    llm_ready = await check_llm_ready_on_startup(max_retries=5, delay=2.0)
    if llm_ready:
        logger.info("✓ LLM service is ready")
    else:
        logger.warning(
            f"⚠ LLM service is not ready. Chat features require model '{REQUIRED_MODEL}'. "
            "The model will be downloaded automatically when the AI service starts."
        )

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
    app.include_router(llm.router)
    app.include_router(scripts.router)
    app.include_router(review.router)
    app.include_router(streaming.router)
    app.include_router(pointcloud.router)
    app.include_router(rosbag.router)
    app.include_router(detect3d.router)
    app.include_router(fusion.router)

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
