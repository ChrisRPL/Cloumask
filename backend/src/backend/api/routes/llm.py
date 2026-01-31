"""LLM service endpoints for language model connectivity."""

import asyncio
import logging

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.llm.config import REQUIRED_MODEL
from backend.api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


class LLMStatus(BaseModel):
    """LLM service status."""

    available: bool = Field(description="Whether LLM service is reachable")
    url: str = Field(description="LLM service API URL")
    error: str | None = Field(default=None, description="Error message if unavailable")


class LLMModel(BaseModel):
    """Information about an LLM model."""

    name: str = Field(description="Model name (e.g., 'qwen3:14b')")
    size: str = Field(description="Model size on disk")
    modified: str = Field(description="Last modified timestamp")


class LLMModelsResponse(BaseModel):
    """Response containing available LLM models."""

    models: list[LLMModel] = Field(description="List of available models")
    default_model: str = Field(description="Configured default model")


class LLMGenerateRequest(BaseModel):
    """Request to generate text with LLM."""

    prompt: str = Field(description="The prompt to generate from")
    model: str | None = Field(default=None, description="Model to use (defaults to configured)")
    stream: bool = Field(default=False, description="Whether to stream the response")


class LLMGenerateResponse(BaseModel):
    """Response from LLM generation."""

    response: str = Field(description="Generated text")
    model: str = Field(description="Model used")
    done: bool = Field(description="Whether generation is complete")


class LLMPullRequest(BaseModel):
    """Request to pull a model."""

    model: str = Field(description="Model name to pull (e.g., 'qwen3:14b')")


class LLMPullStatus(BaseModel):
    """Status of a model pull operation."""

    model: str = Field(description="Model being pulled")
    status: str = Field(description="Current status (pulling, success, error)")
    progress: float = Field(default=0.0, description="Download progress 0-100")
    message: str | None = Field(default=None, description="Status message")


class LLMReadyResponse(BaseModel):
    """Response from ensure-ready check."""

    ready: bool = Field(description="Whether LLM is ready with required model")
    service_running: bool = Field(description="Whether LLM service is running")
    required_model: str = Field(description="The required model name")
    model_available: bool = Field(description="Whether the required model is available")
    error: str | None = Field(default=None, description="Error message if not ready")


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@router.get("/status", response_model=LLMStatus)
async def get_llm_status() -> LLMStatus:
    """
    Check if LLM service is available and responding.

    Returns the status of the LLM service.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            if response.status_code == 200:
                return LLMStatus(
                    available=True,
                    url=settings.ollama_host,
                )
            else:
                return LLMStatus(
                    available=False,
                    url=settings.ollama_host,
                    error=f"LLM service returned status {response.status_code}",
                )
    except httpx.ConnectError:
        return LLMStatus(
            available=False,
            url=settings.ollama_host,
            error="Cannot connect to LLM service. Is it running?",
        )
    except Exception as e:
        return LLMStatus(
            available=False,
            url=settings.ollama_host,
            error=str(e),
        )


@router.get("/models", response_model=LLMModelsResponse)
async def list_llm_models() -> LLMModelsResponse:
    """
    List available LLM models.

    Returns all models available in the LLM service.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            response.raise_for_status()
            data = response.json()

        models = []
        for model in data.get("models", []):
            models.append(
                LLMModel(
                    name=model.get("name", "unknown"),
                    size=_format_size(model.get("size", 0)),
                    modified=model.get("modified_at", "unknown"),
                )
            )

        return LLMModelsResponse(
            models=models,
            default_model=settings.ollama_model,
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LLM service. Is it running?",
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="LLM service request timed out",
        ) from None
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LLM service error: {e.response.text}",
        ) from e


@router.post("/generate", response_model=LLMGenerateResponse)
async def generate_text(request: LLMGenerateRequest) -> LLMGenerateResponse:
    """
    Generate text using the LLM.

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

        return LLMGenerateResponse(
            response=data.get("response", ""),
            model=model,
            done=data.get("done", True),
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LLM service",
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="LLM generation timed out",
        ) from None
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LLM service error: {e.response.text}",
        ) from e


async def _check_model_exists(model_name: str) -> bool:
    """Check if a model exists in the LLM service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            if response.status_code != 200:
                return False
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check for exact match or match without tag
            for m in models:
                if m == model_name or m.startswith(f"{model_name}:"):
                    return True
                # Handle case where model_name has tag but stored without
                if model_name.split(":")[0] == m.split(":")[0]:
                    return True
            return False
    except Exception:
        return False


async def _pull_model_stream(model_name: str):
    """Stream model pull progress."""
    async with (
        httpx.AsyncClient(timeout=None) as client,
        client.stream(
            "POST",
            f"{settings.ollama_host}/api/pull",
            json={"name": model_name},
        ) as response,
    ):
        async for line in response.aiter_lines():
            if line:
                yield f"data: {line}\n\n"


@router.post("/pull", response_model=LLMPullStatus)
async def pull_model(request: LLMPullRequest) -> LLMPullStatus:
    """
    Pull a model from the model registry.

    This starts a model download. For large models, use /pull/stream for progress.
    """
    model_name = request.model

    # Check if model already exists
    if await _check_model_exists(model_name):
        return LLMPullStatus(
            model=model_name,
            status="success",
            progress=100.0,
            message="Model already available",
        )

    try:
        # Start the pull (non-streaming, waits for completion)
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{settings.ollama_host}/api/pull",
                json={"name": model_name, "stream": False},
            )
            response.raise_for_status()

        return LLMPullStatus(
            model=model_name,
            status="success",
            progress=100.0,
            message="Model pulled successfully",
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LLM service. Is it running?",
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Model pull timed out (>10 minutes)",
        ) from None
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LLM service error: {e.response.text}",
        ) from e


@router.post("/pull/stream")
async def pull_model_stream(request: LLMPullRequest) -> StreamingResponse:
    """
    Pull a model with streaming progress updates.

    Returns Server-Sent Events with pull progress.
    """
    return StreamingResponse(
        _pull_model_stream(request.model),
        media_type="text/event-stream",
    )


@router.get("/ensure-ready", response_model=LLMReadyResponse)
async def ensure_ready() -> LLMReadyResponse:
    """
    Check if LLM service is ready with the required model.

    Returns status of LLM service and whether the required model is available.
    Does NOT automatically pull the model - use /pull for that.
    """
    # Check LLM status
    status = await get_llm_status()

    if not status.available:
        return LLMReadyResponse(
            ready=False,
            service_running=False,
            required_model=REQUIRED_MODEL,
            model_available=False,
            error=status.error or "LLM service is not running",
        )

    # Check if required model exists
    model_available = await _check_model_exists(REQUIRED_MODEL)

    if not model_available:
        return LLMReadyResponse(
            ready=False,
            service_running=True,
            required_model=REQUIRED_MODEL,
            model_available=False,
            error=f"Required model '{REQUIRED_MODEL}' is not installed",
        )

    return LLMReadyResponse(
        ready=True,
        service_running=True,
        required_model=REQUIRED_MODEL,
        model_available=True,
    )


@router.post("/ensure-ready", response_model=LLMReadyResponse)
async def ensure_ready_with_pull() -> LLMReadyResponse:
    """
    Ensure LLM service is ready, pulling the required model if needed.

    This will:
    1. Check if LLM service is running
    2. Check if the required model is available
    3. Pull the model if not available (may take several minutes)
    """
    # Check LLM status
    status = await get_llm_status()

    if not status.available:
        return LLMReadyResponse(
            ready=False,
            service_running=False,
            required_model=REQUIRED_MODEL,
            model_available=False,
            error=status.error or "LLM service is not running. Please start the AI service first.",
        )

    # Check if required model exists
    model_available = await _check_model_exists(REQUIRED_MODEL)

    if not model_available:
        logger.info(f"Required model '{REQUIRED_MODEL}' not found, pulling...")
        try:
            # Pull the model (this can take a while)
            async with httpx.AsyncClient(timeout=1800.0) as client:  # 30 min timeout
                response = await client.post(
                    f"{settings.ollama_host}/api/pull",
                    json={"name": REQUIRED_MODEL, "stream": False},
                )
                response.raise_for_status()

            logger.info(f"Successfully pulled model '{REQUIRED_MODEL}'")
            model_available = True

        except httpx.TimeoutException:
            return LLMReadyResponse(
                ready=False,
                service_running=True,
                required_model=REQUIRED_MODEL,
                model_available=False,
                error=f"Timed out downloading model '{REQUIRED_MODEL}'",
            )
        except Exception as e:
            return LLMReadyResponse(
                ready=False,
                service_running=True,
                required_model=REQUIRED_MODEL,
                model_available=False,
                error=f"Failed to download model: {e}",
            )

    return LLMReadyResponse(
        ready=True,
        service_running=True,
        required_model=REQUIRED_MODEL,
        model_available=True,
    )


async def check_llm_ready_on_startup(max_retries: int = 5, delay: float = 2.0) -> bool:
    """
    Check if LLM service is ready during app startup.

    Retries several times to allow for LLM service to start.
    Returns True if ready, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            status = await get_llm_status()
            if status.available:
                model_exists = await _check_model_exists(REQUIRED_MODEL)
                if model_exists:
                    logger.info(f"LLM service ready with model '{REQUIRED_MODEL}'")
                    return True
                else:
                    logger.warning(
                        f"LLM service running but model '{REQUIRED_MODEL}' not found. "
                        f"The model will be downloaded on first use."
                    )
                    return False
            else:
                logger.warning(
                    f"LLM service not available (attempt {attempt + 1}/{max_retries}): "
                    f"{status.error}"
                )
        except Exception as e:
            logger.warning(f"Error checking LLM service (attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(delay)

    logger.error("LLM service not available after all retries")
    return False
