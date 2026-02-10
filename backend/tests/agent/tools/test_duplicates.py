"""Tests for find_duplicates agent tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.agent.tools.duplicates import FindDuplicatesTool
from backend.data.duplicates import DuplicateGroup, DuplicateResult


def _make_duplicate_result(
    image_a: Path,
    image_b: Path,
    *,
    method: str = "phash",
    threshold: float = 0.9,
) -> DuplicateResult:
    """Build a duplicate detection result fixture."""
    return DuplicateResult(
        groups=[
            DuplicateGroup(
                images=[image_a, image_b],
                similarity_scores=[0.98],
            )
        ],
        total_images=2,
        method=method,
        threshold=threshold,
    )


class TestFindDuplicatesTool:
    @pytest.mark.asyncio
    async def test_find_duplicates_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should return grouped duplicate paths on success."""
        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        img1.write_bytes(b"a")
        img2.write_bytes(b"b")

        calls: dict[str, Any] = {}

        def _fake_detect(
            paths: list[Path],
            *,
            method: str,
            threshold: float,
            progress_callback: Any = None,
        ) -> DuplicateResult:
            calls["paths"] = list(paths)
            calls["method"] = method
            calls["threshold"] = threshold
            calls["progress_callback"] = progress_callback
            return _make_duplicate_result(img1, img2, method=method, threshold=threshold)

        monkeypatch.setattr("backend.agent.tools.duplicates.detect_duplicates", _fake_detect)

        tool = FindDuplicatesTool()
        result = await tool.run(
            path=str(tmp_path),
            method="phash",
            threshold=0.95,
            max_groups=10,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["total_images"] == 2
        assert result.data["num_groups"] == 1
        assert result.data["num_duplicates"] == 1
        assert result.data["truncated"] is False
        assert result.data["removed"] == []
        assert result.data["groups"][0]["representative"] == str(img1)
        assert result.data["groups"][0]["duplicates"] == [str(img2)]
        assert callable(calls["progress_callback"])
        assert set(calls["paths"]) == {img1, img2}
        assert calls["method"] == "phash"
        assert calls["threshold"] == 0.95

    @pytest.mark.asyncio
    async def test_find_duplicates_returns_empty_result_when_no_images(self, tmp_path: Path) -> None:
        """Should return success with zero groups when no images are present."""
        tool = FindDuplicatesTool()
        result = await tool.run(path=str(tmp_path))

        assert result.success is True
        assert result.data is not None
        assert result.data["total_images"] == 0
        assert result.data["num_groups"] == 0
        assert result.data["num_duplicates"] == 0
        assert result.data["groups"] == []

    @pytest.mark.asyncio
    async def test_find_duplicates_auto_remove(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """auto_remove=True should delete duplicate files."""
        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        img1.write_bytes(b"a")
        img2.write_bytes(b"b")

        def _fake_detect(
            paths: list[Path],
            *,
            method: str,
            threshold: float,
            progress_callback: Any = None,
        ) -> DuplicateResult:
            return _make_duplicate_result(img1, img2, method=method, threshold=threshold)

        monkeypatch.setattr("backend.agent.tools.duplicates.detect_duplicates", _fake_detect)

        tool = FindDuplicatesTool()
        result = await tool.run(path=str(tmp_path), auto_remove=True)

        assert result.success is True
        assert result.data is not None
        assert result.data["removed"] == [str(img2)]
        assert result.data["remove_errors"] == []
        assert img1.exists() is True
        assert img2.exists() is False

    @pytest.mark.asyncio
    async def test_find_duplicates_invalid_threshold(self, tmp_path: Path) -> None:
        """Threshold must be in [0, 1]."""
        tool = FindDuplicatesTool()
        result = await tool.run(path=str(tmp_path), threshold=1.5)

        assert result.success is False
        assert result.error is not None
        assert "threshold" in result.error.lower()

    @pytest.mark.asyncio
    async def test_find_duplicates_invalid_method(self, tmp_path: Path) -> None:
        """Method should be validated against supported values."""
        tool = FindDuplicatesTool()
        result = await tool.run(path=str(tmp_path), method="not-a-method")

        assert result.success is False
        assert result.error is not None
        assert "method" in result.error.lower()

    @pytest.mark.asyncio
    async def test_find_duplicates_invalid_max_groups(self, tmp_path: Path) -> None:
        """max_groups must be positive."""
        tool = FindDuplicatesTool()
        result = await tool.run(path=str(tmp_path), max_groups=0)

        assert result.success is False
        assert result.error is not None
        assert "max_groups" in result.error.lower()

    @pytest.mark.asyncio
    async def test_find_duplicates_missing_path(self, tmp_path: Path) -> None:
        """Non-existent input path should fail."""
        tool = FindDuplicatesTool()
        result = await tool.run(path=str(tmp_path / "missing"))

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_find_duplicates_runtime_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Runtime errors from backend detection should become tool errors."""
        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        img1.write_bytes(b"a")
        img2.write_bytes(b"b")

        def _raise_runtime(
            paths: list[Path],
            *,
            method: str,
            threshold: float,
            progress_callback: Any = None,
        ) -> DuplicateResult:
            raise RuntimeError("CLIP dependencies are missing")

        monkeypatch.setattr("backend.agent.tools.duplicates.detect_duplicates", _raise_runtime)

        tool = FindDuplicatesTool()
        result = await tool.run(path=str(tmp_path), method="clip")

        assert result.success is False
        assert result.error is not None
        assert "duplicate detection failed" in result.error.lower()


class TestFindDuplicatesToolRegistration:
    def test_find_duplicates_tool_registered(self) -> None:
        """find_duplicates should be available in the global tool registry."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        initialize_tools()
        registry = get_tool_registry()

        assert registry.has("find_duplicates")
        tool = registry.get("find_duplicates")
        assert tool is not None

        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "find_duplicates"
        assert "path" in properties
        assert "method" in properties
        assert "threshold" in properties
        assert "auto_remove" in properties
        assert "max_groups" in properties
