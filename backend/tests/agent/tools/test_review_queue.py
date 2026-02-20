"""Tests for ReviewQueueTool annotation import behavior."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from backend.agent.tools.review_queue import ReviewQueueTool
from backend.api.routes import review as review_routes

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9fY0QAAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def clear_review_items() -> None:
    """Avoid state leak between tests."""
    review_routes._review_items.clear()
    yield
    review_routes._review_items.clear()


@pytest.mark.asyncio
async def test_review_queue_reads_dataset_root_and_prefixed_yolo_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Should resolve train/labels + data.yaml and map prefixed stems."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "scene_010.png"
    image_path.write_bytes(_PNG_BYTES)

    source_path = tmp_path / "detections_yolo"
    labels_dir = source_path / "train" / "labels"
    labels_dir.mkdir(parents=True)
    (source_path / "data.yaml").write_text(
        "train: train/images\n"
        "val: val/images\n"
        "names: [person, pallet]\n",
        encoding="utf-8",
    )
    (labels_dir / "00000_scene_010.txt").write_text(
        "1 0.5 0.5 0.2 0.4 0.88\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(review_routes, "_save_to_disk", lambda _execution_id: None)

    tool = ReviewQueueTool()
    result = await tool.run(
        source_path=str(source_path),
        image_dir=str(image_dir),
        execution_id="exec_test",
        project_id="project_test",
    )

    assert result.success
    assert result.data is not None
    assert result.data["created_count"] == 1

    assert len(review_routes._review_items) == 1
    review_item = next(iter(review_routes._review_items.values()))
    assert review_item.execution_id == "exec_test"
    assert review_item.project_id == "project_test"
    assert review_item.file_name == image_path.name
    assert len(review_item.annotations) == 1

    annotation = review_item.annotations[0]
    assert annotation.label == "pallet"
    assert annotation.confidence == pytest.approx(0.88)
    assert annotation.bbox.x == pytest.approx(0.4)
    assert annotation.bbox.y == pytest.approx(0.3)
    assert annotation.bbox.width == pytest.approx(0.2)
    assert annotation.bbox.height == pytest.approx(0.4)
