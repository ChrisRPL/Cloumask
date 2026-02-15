"""Integration tests for review API routes."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes.review import _review_items

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9fY0QAAAAASUVORK5CYII="
)


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_review_items() -> None:
    """Avoid state leak between tests."""
    _review_items.clear()
    yield
    _review_items.clear()


def test_get_local_image_success(client: TestClient, tmp_path: Path) -> None:
    """Should return an image file for valid local image path."""
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(_PNG_BYTES)

    response = client.get("/api/review/image", params={"path": str(image_path)})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 0


def test_get_local_image_not_found(client: TestClient, tmp_path: Path) -> None:
    """Should return 404 when image does not exist."""
    missing = tmp_path / "missing.png"

    response = client.get("/api/review/image", params={"path": str(missing)})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_local_image_rejects_non_image_extension(
    client: TestClient, tmp_path: Path
) -> None:
    """Should return 415 for unsupported file extensions."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text("not an image")

    response = client.get("/api/review/image", params={"path": str(text_file)})

    assert response.status_code == 415
    assert "unsupported image extension" in response.json()["detail"].lower()
