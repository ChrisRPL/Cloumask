"""Tests for duplicate and near-duplicate image detection."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.data.duplicates import (
    DuplicateDetector,
    DuplicateResult,
    PerceptualHasher,
    find_duplicates,
)


def _create_base_image(size: int = 96) -> np.ndarray:
    """Create a deterministic non-trivial pattern image."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = (20, 40, 20)
    image[:, size // 3 : (2 * size) // 3] = (60, 120, 60)
    image[size // 4 : (3 * size) // 4, size // 4 : (3 * size) // 4] = (230, 230, 230)
    image[np.arange(size), np.arange(size)] = (255, 0, 0)
    image[np.arange(size), np.arange(size - 1, -1, -1)] = (0, 0, 255)
    return image


def _create_different_image(size: int = 96) -> np.ndarray:
    """Create a very different deterministic pattern image."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[::2, :] = (245, 245, 245)
    image[:, ::3] = (20, 160, 220)
    image[size // 3 : (2 * size) // 3, :] = (15, 15, 15)
    return image


@pytest.fixture
def sample_images(tmp_path: Path) -> dict[str, Path]:
    """Create exact duplicate, near duplicate, and different images."""
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    img3_path = tmp_path / "img3.png"
    img4_path = tmp_path / "img4.png"

    img1 = _create_base_image()
    Image.fromarray(img1).save(img1_path)
    shutil.copy2(img1_path, img2_path)

    img3 = img1.copy()
    img3[34:58, 34:58] = np.clip(img3[34:58, 34:58].astype(np.int16) + 12, 0, 255).astype(
        np.uint8
    )
    Image.fromarray(img3).save(img3_path)

    Image.fromarray(_create_different_image()).save(img4_path)

    return {
        "img1": img1_path,
        "img2": img2_path,
        "img3": img3_path,
        "img4": img4_path,
    }


class TestPerceptualHasher:
    """Tests for perceptual hash algorithms."""

    @pytest.mark.parametrize("method_name", ["ahash", "dhash", "phash"])
    def test_hash_methods_return_values(
        self,
        sample_images: dict[str, Path],
        method_name: str,
    ) -> None:
        """Hash methods should produce integer hashes for valid images."""
        hasher = PerceptualHasher(hash_size=8)
        method = getattr(hasher, method_name)
        value = method(sample_images["img1"])
        assert isinstance(value, int)

    def test_exact_duplicates_have_same_dhash(self, sample_images: dict[str, Path]) -> None:
        """Exact duplicates should have identical dHash."""
        hasher = PerceptualHasher(hash_size=8)
        hash1 = hasher.dhash(sample_images["img1"])
        hash2 = hasher.dhash(sample_images["img2"])
        assert hash1 == hash2

    def test_hamming_distance_identical_is_zero(self, sample_images: dict[str, Path]) -> None:
        """Hamming distance must be zero for identical hashes."""
        hasher = PerceptualHasher(hash_size=8)
        hash1 = hasher.phash(sample_images["img1"])
        hash2 = hasher.phash(sample_images["img2"])
        assert hash1 is not None
        assert hash2 is not None
        assert hasher.hamming_distance(hash1, hash2) == 0

    def test_hamming_distance_different_is_nonzero(self, sample_images: dict[str, Path]) -> None:
        """Different images should have a non-zero hash distance."""
        hasher = PerceptualHasher(hash_size=8)
        hash1 = hasher.ahash(sample_images["img1"])
        hash4 = hasher.ahash(sample_images["img4"])
        assert hash1 is not None
        assert hash4 is not None
        assert hasher.hamming_distance(hash1, hash4) > 0


class TestDuplicateDetector:
    """Tests for duplicate grouping logic."""

    def test_find_exact_duplicates_phash(self, sample_images: dict[str, Path]) -> None:
        """pHash should group exact duplicates."""
        detector = DuplicateDetector(method="phash")
        result = detector.find_duplicates(list(sample_images.values()), threshold=0.98)

        assert isinstance(result, DuplicateResult)
        assert result.num_groups >= 1
        assert any(
            {sample_images["img1"], sample_images["img2"]}.issubset(set(group.images))
            for group in result.groups
        )

    def test_find_near_duplicates_dhash(self, sample_images: dict[str, Path]) -> None:
        """dHash should group near-duplicate images at moderate threshold."""
        detector = DuplicateDetector(method="dhash")
        result = detector.find_duplicates(list(sample_images.values()), threshold=0.85)

        assert result.num_groups >= 1
        assert any(
            {sample_images["img1"], sample_images["img2"], sample_images["img3"]}.issubset(
                set(group.images)
            )
            for group in result.groups
        )

    def test_different_images_not_grouped(self, sample_images: dict[str, Path]) -> None:
        """Very different images should not be grouped with strict threshold."""
        detector = DuplicateDetector(method="phash")
        result = detector.find_duplicates(
            [sample_images["img1"], sample_images["img4"]],
            threshold=0.95,
        )
        assert result.num_groups == 0

    def test_get_duplicates_to_remove(self, sample_images: dict[str, Path]) -> None:
        """Duplicate removal list should exclude representative images."""
        result = find_duplicates(list(sample_images.values()), method="phash", threshold=0.95)
        to_remove = result.get_duplicates_to_remove()
        assert sample_images["img1"] not in to_remove
        assert sample_images["img2"] in to_remove

    def test_progress_callback_invoked(self, sample_images: dict[str, Path]) -> None:
        """Progress callback should be called during hashing/comparison."""
        callbacks: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, message: str) -> None:
            callbacks.append((current, total, message))

        detector = DuplicateDetector(method="ahash", progress_callback=callback)
        detector.find_duplicates(list(sample_images.values()), threshold=0.9)

        assert callbacks
        assert any(total >= current >= 1 for current, total, _ in callbacks if total > 0)

    def test_clip_similarity_path_supported(
        self,
        sample_images: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLIP flow should cluster groups based on cosine similarity."""
        detector = DuplicateDetector(method="clip")
        p1, p2, p4 = sample_images["img1"], sample_images["img2"], sample_images["img4"]

        fake_embeddings = {
            p1: np.array([1.0, 0.0], dtype=np.float32),
            p2: np.array([0.98, 0.2], dtype=np.float32),
            p4: np.array([0.0, 1.0], dtype=np.float32),
        }
        fake_embeddings = {
            path: emb / np.linalg.norm(emb)
            for path, emb in fake_embeddings.items()
        }

        monkeypatch.setattr(detector, "_compute_embeddings", lambda paths: fake_embeddings)
        result = detector.find_duplicates([p1, p2, p4], threshold=0.95)

        assert result.num_groups == 1
        assert {p1, p2}.issubset(set(result.groups[0].images))

    def test_invalid_method_raises(self) -> None:
        """Unknown methods should be rejected."""
        with pytest.raises(ValueError, match="Unknown method"):
            DuplicateDetector(method="not-a-method")

    def test_invalid_threshold_raises(self, sample_images: dict[str, Path]) -> None:
        """Threshold must be between 0 and 1."""
        detector = DuplicateDetector(method="phash")
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            detector.find_duplicates(list(sample_images.values()), threshold=1.5)

    def test_near_duplicate_recall_meets_target(self, tmp_path: Path) -> None:
        """Near-duplicate detection should recover at least 95% of synthetic variants."""
        base_path = tmp_path / "base.png"
        base_image = _create_base_image()
        Image.fromarray(base_image).save(base_path)

        near_duplicate_paths: list[Path] = []
        for idx in range(20):
            variant = base_image.copy()
            patch_y = 8 + (idx % 5) * 12
            patch_x = 8 + (idx // 5) * 12
            patch = variant[patch_y : patch_y + 6, patch_x : patch_x + 6].astype(np.int16)
            patch = np.clip(patch + ((idx % 3) - 1) * 8, 0, 255).astype(np.uint8)
            variant[patch_y : patch_y + 6, patch_x : patch_x + 6] = patch

            near_path = tmp_path / f"near_{idx:02d}.png"
            Image.fromarray(variant).save(near_path)
            near_duplicate_paths.append(near_path)

        distractor_path = tmp_path / "distractor.png"
        Image.fromarray(_create_different_image()).save(distractor_path)

        detector = DuplicateDetector(method="dhash")
        result = detector.find_duplicates(
            [base_path, *near_duplicate_paths, distractor_path],
            threshold=0.85,
        )

        recovered: set[Path] = set()
        for group in result.groups:
            if base_path in group.images:
                recovered.update(path for path in group.images if path != base_path)

        recall = len(recovered & set(near_duplicate_paths)) / len(near_duplicate_paths)
        assert recall >= 0.95
