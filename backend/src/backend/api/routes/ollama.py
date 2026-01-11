"""Ollama integration endpoints for LLM connectivity."""

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


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


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
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out",
        ) from None
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama error: {e.response.text}",
        ) from e


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
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama generation timed out",
        ) from None
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama error: {e.response.text}",
        ) from e
