"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend import __version__
from backend.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        """Health response should contain required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data
        assert "backend_src_path" in data

    def test_health_status_is_healthy(self, client: TestClient) -> None:
        """Health status should be 'healthy' when server is running."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_version_matches(self, client: TestClient) -> None:
        """Health version should match package version."""
        response = client.get("/health")
        data = response.json()

        assert data["version"] == __version__


class TestReadyEndpoint:
    """Tests for /ready endpoint."""

    def test_ready_returns_200(self, client: TestClient) -> None:
        """Ready endpoint should return 200 OK."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_is_true(self, client: TestClient) -> None:
        """Ready should be True when server is running."""
        response = client.get("/ready")
        data = response.json()

        assert data["ready"] is True

    def test_ready_checks_passed(self, client: TestClient) -> None:
        """All ready checks should pass."""
        response = client.get("/ready")
        data = response.json()

        assert all(data["checks"].values())
        assert "backend_src_path" in data


class TestRootEndpoint:
    """Tests for root / endpoint."""

    def test_root_returns_200(self, client: TestClient) -> None:
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_name(self, client: TestClient) -> None:
        """Root response should contain app name."""
        response = client.get("/")
        data = response.json()

        assert data["name"] == "Cloumask Backend"

    def test_root_contains_docs_link(self, client: TestClient) -> None:
        """Root response should contain docs link."""
        response = client.get("/")
        data = response.json()

        assert data["docs"] == "/docs"
