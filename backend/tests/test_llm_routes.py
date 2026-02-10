"""Tests for /llm endpoints."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import llm as llm_routes


def _available_status() -> llm_routes.LLMStatus:
    return llm_routes.LLMStatus(available=True, url="http://localhost:11434", error=None)


def test_ensure_ready_uses_configured_model(monkeypatch) -> None:
    """GET /llm/ensure-ready should report the configured required model."""
    client = TestClient(app)
    monkeypatch.setattr(llm_routes.settings, "ollama_model", "qwen3:8b")

    async def fake_status() -> llm_routes.LLMStatus:
        return _available_status()

    async def fake_exists(model_name: str) -> bool:
        assert model_name == "qwen3:8b"
        return False

    monkeypatch.setattr(llm_routes, "get_llm_status", fake_status)
    monkeypatch.setattr(llm_routes, "_check_model_exists", fake_exists)

    response = client.get("/llm/ensure-ready")
    data = response.json()

    assert response.status_code == 200
    assert data["required_model"] == "qwen3:8b"
    assert data["model_available"] is False


def test_ensure_ready_with_pull_uses_configured_model(monkeypatch) -> None:
    """POST /llm/ensure-ready should pull the model from settings.ollama_model."""
    client = TestClient(app)
    monkeypatch.setattr(llm_routes.settings, "ollama_model", "mistral:7b-instruct")

    async def fake_status() -> llm_routes.LLMStatus:
        return _available_status()

    async def fake_exists(_model_name: str) -> bool:
        return False

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None
    post_mock = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(llm_routes, "get_llm_status", fake_status)
    monkeypatch.setattr(llm_routes, "_check_model_exists", fake_exists)
    monkeypatch.setattr("httpx.AsyncClient.post", post_mock)

    response = client.post("/llm/ensure-ready")
    data = response.json()

    assert response.status_code == 200
    assert data["ready"] is True
    assert data["required_model"] == "mistral:7b-instruct"
    assert post_mock.await_count == 1
    assert post_mock.await_args.kwargs["json"]["name"] == "mistral:7b-instruct"
