"""Health check endpoint for sidecar status verification."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend import __version__

router = APIRouter()
BACKEND_SRC_PATH = str(Path(__file__).resolve().parents[3])


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Current health status of the sidecar"
    )
    version: str = Field(description="Backend version")
    timestamp: str = Field(description="ISO 8601 timestamp")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components",
    )
    backend_src_path: str = Field(
        description="Absolute path to backend/src for sidecar ownership checks"
    )


class ReadyResponse(BaseModel):
    """Response model for readiness check."""

    ready: bool = Field(description="Whether the sidecar is ready to accept requests")
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual readiness checks",
    )
    backend_src_path: str = Field(
        description="Absolute path to backend/src for sidecar ownership checks"
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
        timestamp=datetime.now(UTC).isoformat(),
        components={
            "api": "healthy",
            "agent": "not_loaded",  # Will be updated in 02-agent-system
            "cv_models": "not_loaded",  # Will be updated in 03-cv-models
        },
        backend_src_path=BACKEND_SRC_PATH,
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    """
    Check if the sidecar is ready to process requests.

    Used for startup probes to ensure sidecar is fully initialized.
    """
    checks = {
        "api_running": True,
        "routes_loaded": True,
    }

    return ReadyResponse(
        ready=all(checks.values()),
        checks=checks,
        backend_src_path=BACKEND_SRC_PATH,
    )


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Cloumask Backend",
        "version": __version__,
        "docs": "/docs",
    }
