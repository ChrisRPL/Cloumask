"""Integration tests for review API routes."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.models.review import (
    Annotation,
    BoundingBox,
    ImageDimensions,
    ReviewItem,
    ReviewStatus,
)
from backend.api.routes import review as review_routes

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
    review_routes._review_items.clear()
    yield
    review_routes._review_items.clear()


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


def test_import_annotations_reads_nested_yolo_labels_and_yaml_names(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Should import labels from YOLO dataset roots and map prefixed stems."""
    execution_id = "exec_test"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "frame_001.png"
    image_path.write_bytes(_PNG_BYTES)

    review_item = ReviewItem(
        id="item-1",
        file_path=str(image_path.resolve()),
        file_name=image_path.name,
        dimensions=ImageDimensions(width=1, height=1),
        thumbnail_url="data:image/png;base64,",
        annotations=[],
        original_annotations=[],
        status=ReviewStatus.PENDING,
        reviewed_at=None,
        flagged=False,
        flag_reason=None,
    )
    review_routes._review_items[review_item.id] = review_item
    monkeypatch.setattr(review_routes, "_save_to_disk", lambda _execution_id: None)

    annotations_root = tmp_path / "detections_yolo"
    labels_dir = annotations_root / "train" / "labels"
    labels_dir.mkdir(parents=True)
    (annotations_root / "data.yaml").write_text(
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: person\n"
        "  1: forklift\n",
        encoding="utf-8",
    )
    (labels_dir / "00000_frame_001.txt").write_text(
        "1 0.5 0.5 0.2 0.4 0.75\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/review/import-annotations",
        params={
            "execution_id": execution_id,
            "annotations_dir": str(annotations_root),
            "image_dir": str(image_dir),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_count"] == 1
    assert payload["total_items"] == 1

    imported = review_routes._review_items["item-1"]
    assert len(imported.annotations) == 1
    annotation = imported.annotations[0]
    assert annotation.label == "forklift"
    assert annotation.confidence == pytest.approx(0.75)
    assert annotation.bbox.x == pytest.approx(0.4)
    assert annotation.bbox.y == pytest.approx(0.3)
    assert annotation.bbox.width == pytest.approx(0.2)
    assert annotation.bbox.height == pytest.approx(0.4)


def test_hitl_annotation_edit_approve_reject_flow(client: TestClient, tmp_path: Path) -> None:
    """Should support edit + approve + reject transitions for review items."""
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(_PNG_BYTES)

    review_item = ReviewItem(
        id="item-flow",
        file_path=str(image_path.resolve()),
        file_name=image_path.name,
        dimensions=ImageDimensions(width=1, height=1),
        thumbnail_url="data:image/png;base64,",
        annotations=[
            Annotation(
                id="ann-1",
                type="bbox",
                label="person",
                confidence=0.8,
                bbox=BoundingBox(x=0.2, y=0.2, width=0.3, height=0.4),
                color="#166534",
                visible=True,
            )
        ],
        original_annotations=[],
        status=ReviewStatus.PENDING,
        reviewed_at=None,
        flagged=False,
        flag_reason=None,
    )
    review_routes._review_items[review_item.id] = review_item

    edit_response = client.put(
        "/api/review/items/item-flow/annotations/ann-1",
        json={
            "label": "forklift",
            "bbox": {"x": 0.25, "y": 0.2, "width": 0.3, "height": 0.4},
        },
    )
    assert edit_response.status_code == 200
    assert edit_response.json()["label"] == "forklift"
    assert review_routes._review_items["item-flow"].status == ReviewStatus.MODIFIED

    approve_response = client.post(
        "/api/review/batch-approve",
        json={"item_ids": ["item-flow"]},
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["success_count"] == 1
    assert review_routes._review_items["item-flow"].status == ReviewStatus.APPROVED

    reject_response = client.post(
        "/api/review/batch-reject",
        json={"item_ids": ["item-flow"]},
    )
    assert reject_response.status_code == 200
    assert reject_response.json()["success_count"] == 1
    assert review_routes._review_items["item-flow"].status == ReviewStatus.REJECTED
