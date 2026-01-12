"""Shared fixtures for agent tool tests.

This module provides common test fixtures used across multiple test files.
"""

import pytest


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset directory with various file types."""
    # Create image directory with files
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        (img_dir / f"image_{i}.jpg").write_bytes(b"fake_image_data" * 100)

    # Create video directory with files
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    for i in range(3):
        (vid_dir / f"video_{i}.mp4").write_bytes(b"fake_video_data" * 1000)

    # Add some annotation files
    anno_dir = tmp_path / "annotations"
    anno_dir.mkdir()
    for i in range(5):
        (anno_dir / f"label_{i}.json").write_text('{"objects": []}')

    return tmp_path


@pytest.fixture
def nested_dataset(tmp_path):
    """Create a nested directory structure."""
    # Level 1
    level1 = tmp_path / "level1"
    level1.mkdir()
    (level1 / "file1.jpg").touch()

    # Level 2
    level2 = level1 / "level2"
    level2.mkdir()
    (level2 / "file2.jpg").touch()

    # Level 3
    level3 = level2 / "level3"
    level3.mkdir()
    (level3 / "file3.jpg").touch()

    return tmp_path


@pytest.fixture
def pointcloud_dataset(tmp_path):
    """Create a point cloud dataset."""
    pc_dir = tmp_path / "pointclouds"
    pc_dir.mkdir()
    for i in range(5):
        (pc_dir / f"scan_{i}.las").touch()
        (pc_dir / f"scan_{i}.pcd").touch()

    return tmp_path


@pytest.fixture
def empty_dataset(tmp_path):
    """Create an empty directory for edge case testing."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    return empty_dir
