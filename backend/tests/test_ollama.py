"""Tests for Ollama integration endpoints."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


class TestOllamaStatusEndpoint:
    """Tests for /ollama/status endpoint."""

    def test_status_returns_200_when_ollama_available(self, client: TestClient) -> None:
        """Status endpoint should return 200 when Ollama is reachable."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/ollama/status")

        assert response.status_code == 200

    def test_status_available_when_ollama_responds(self, client: TestClient) -> None:
        """Status should show available=True when Ollama responds with 200."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/ollama/status")
            data = response.json()

        assert data["available"] is True
        assert data["error"] is None

    def test_status_unavailable_on_connection_error(self, client: TestClient) -> None:
        """Status should show available=False when Ollama is unreachable."""
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
            response = client.get("/ollama/status")
            data = response.json()

        assert response.status_code == 200
        assert data["available"] is False
        assert "Cannot connect" in data["error"]

    def test_status_contains_url(self, client: TestClient) -> None:
        """Status response should contain Ollama URL."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/ollama/status")
            data = response.json()

        assert "url" in data
        assert "localhost:11434" in data["url"]


class TestOllamaModelsEndpoint:
    """Tests for /ollama/models endpoint."""

    def test_models_returns_200_with_model_list(self, client: TestClient) -> None:
        """Models endpoint should return 200 with model list."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json = lambda: {
            "models": [
                {"name": "qwen3:14b", "size": 8000000000, "modified_at": "2025-01-01T00:00:00Z"},
                {"name": "llama3.1:8b", "size": 4000000000, "modified_at": "2025-01-02T00:00:00Z"},
            ]
        }

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/ollama/models")

        assert response.status_code == 200

    def test_models_response_structure(self, client: TestClient) -> None:
        """Models response should contain required fields."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json = lambda: {
            "models": [
                {"name": "qwen3:14b", "size": 8000000000, "modified_at": "2025-01-01T00:00:00Z"},
            ]
        }

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/ollama/models")
            data = response.json()

        assert "models" in data
        assert "default_model" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "qwen3:14b"

    def test_models_returns_503_on_connection_error(self, client: TestClient) -> None:
        """Models endpoint should return 503 when Ollama is unreachable."""
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
            response = client.get("/ollama/models")

        assert response.status_code == 503
        assert "Cannot connect" in response.json()["detail"]

    def test_models_formats_size_correctly(self, client: TestClient) -> None:
        """Model sizes should be formatted as human-readable."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json = lambda: {
            "models": [
                {"name": "test-model", "size": 1073741824, "modified_at": "2025-01-01T00:00:00Z"},  # 1 GB
            ]
        }

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/ollama/models")
            data = response.json()

        assert data["models"][0]["size"] == "1.0 GB"


class TestOllamaGenerateEndpoint:
    """Tests for /ollama/generate endpoint."""

    def test_generate_returns_200_with_response(self, client: TestClient) -> None:
        """Generate endpoint should return 200 with generated text."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json = lambda: {
            "response": "Hello! How can I help you?",
            "model": "qwen3:14b",
            "done": True,
        }

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/ollama/generate",
                json={"prompt": "Hello"},
            )

        assert response.status_code == 200

    def test_generate_response_structure(self, client: TestClient) -> None:
        """Generate response should contain required fields."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json = lambda: {
            "response": "Test response",
            "model": "qwen3:14b",
            "done": True,
        }

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/ollama/generate",
                json={"prompt": "Test prompt"},
            )
            data = response.json()

        assert "response" in data
        assert "model" in data
        assert "done" in data

    def test_generate_uses_provided_model(self, client: TestClient) -> None:
        """Generate should use the model specified in request."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json = lambda: {
            "response": "Response from llama",
            "model": "llama3.1:8b",
            "done": True,
        }

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/ollama/generate",
                json={"prompt": "Hello", "model": "llama3.1:8b"},
            )
            data = response.json()

        assert data["model"] == "llama3.1:8b"

    def test_generate_returns_503_on_connection_error(self, client: TestClient) -> None:
        """Generate endpoint should return 503 when Ollama is unreachable."""
        with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
            response = client.post(
                "/ollama/generate",
                json={"prompt": "Hello"},
            )

        assert response.status_code == 503
