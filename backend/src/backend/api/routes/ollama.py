"""Backward-compatible Ollama API routes.

This module keeps legacy `/ollama/*` endpoints available while the primary
implementation lives under `/llm/*`.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.routes import llm
from backend.api.routes.llm import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    LLMModelsResponse,
    LLMStatus,
)

router = APIRouter(prefix="/ollama", tags=["Ollama"])


@router.get("/status", response_model=LLMStatus)
async def get_ollama_status() -> LLMStatus:
    """Compatibility alias for `GET /llm/status`."""
    return await llm.get_llm_status()


@router.get("/models", response_model=LLMModelsResponse)
async def list_ollama_models() -> LLMModelsResponse:
    """Compatibility alias for `GET /llm/models`."""
    return await llm.list_llm_models()


@router.post("/generate", response_model=LLMGenerateResponse)
async def generate_ollama_text(request: LLMGenerateRequest) -> LLMGenerateResponse:
    """Compatibility alias for `POST /llm/generate`."""
    return await llm.generate_text(request)

