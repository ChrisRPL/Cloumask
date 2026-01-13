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


@pytest.fixture
def temp_image(tmp_path):
    """
    Create a minimal valid RGB image file.

    Returns path to a small but valid JPEG image that can be loaded by CV models.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not installed, skipping image fixture")

    img = Image.new("RGB", (640, 480), color=(128, 64, 32))
    # Add some variation so it's not completely uniform
    pixels = img.load()
    for x in range(0, 640, 10):
        for y in range(0, 480, 10):
            pixels[x, y] = (255, 0, 0)  # type: ignore[index]

    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


@pytest.fixture
def temp_pointcloud(tmp_path):
    """
    Create a minimal point cloud in KITTI binary format.

    Returns path to a binary file with random 3D points (N x 4: x, y, z, intensity).
    """
    import numpy as np

    # Create 1000 random points with xyz and intensity
    points = np.random.rand(1000, 4).astype(np.float32)
    # Scale to reasonable range: x/y: -50 to 50m, z: -2 to 2m
    points[:, :2] = points[:, :2] * 100 - 50  # x, y
    points[:, 2] = points[:, 2] * 4 - 2  # z
    points[:, 3] = points[:, 3]  # intensity 0-1

    path = tmp_path / "test_pointcloud.bin"
    points.tofile(path)
    return path


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create directory with multiple valid images."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not installed, skipping image directory fixture")

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(5):
        img = Image.new("RGB", (320, 240), color=(i * 50, 100, 150))
        (img_dir / f"image_{i}.jpg").parent.mkdir(parents=True, exist_ok=True)
        img.save(img_dir / f"image_{i}.jpg")

    return img_dir
