# Duplicate Detection

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models
> **Blocks:** 23-find-duplicates-tool

## Objective

Implement duplicate and near-duplicate image detection using perceptual hashing and CLIP embeddings to find similar images in datasets.

## Acceptance Criteria

- [ ] Compute perceptual hashes (pHash, dHash, aHash)
- [ ] Compare images using hash Hamming distance
- [ ] Support CLIP embedding similarity
- [ ] Configurable similarity threshold
- [ ] Cluster duplicates into groups
- [ ] Return results with similarity scores
- [ ] Unit tests with known duplicates

## Implementation Steps

### 1. Create duplicates.py

Create `backend/data/duplicates.py`:

```python
"""Duplicate and near-duplicate image detection.

Uses perceptual hashing and embedding similarity to find
similar or duplicate images in a dataset.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """A group of duplicate/similar images.

    Attributes:
        images: List of image paths in this group
        similarity_scores: Pairwise similarity scores
        representative: Path to representative image (first found)
    """
    images: list[Path]
    similarity_scores: list[float] = field(default_factory=list)

    @property
    def representative(self) -> Path:
        """First image in group as representative."""
        return self.images[0] if self.images else Path()

    @property
    def duplicates(self) -> list[Path]:
        """All images except the representative."""
        return self.images[1:] if len(self.images) > 1 else []

    @property
    def count(self) -> int:
        """Number of images in group."""
        return len(self.images)


@dataclass
class DuplicateResult:
    """Result of duplicate detection.

    Attributes:
        groups: List of duplicate groups
        total_images: Total images processed
        hash_type: Hash algorithm used
        threshold: Similarity threshold
    """
    groups: list[DuplicateGroup]
    total_images: int
    hash_type: str
    threshold: float

    @property
    def num_duplicates(self) -> int:
        """Total number of duplicate images (excluding representatives)."""
        return sum(g.count - 1 for g in self.groups)

    @property
    def num_groups(self) -> int:
        """Number of duplicate groups."""
        return len(self.groups)

    def get_duplicates_to_remove(self) -> list[Path]:
        """Get list of duplicate images to remove (keep representatives)."""
        return [p for g in self.groups for p in g.duplicates]


class PerceptualHasher:
    """Compute perceptual hashes for images."""

    def __init__(self, hash_size: int = 8) -> None:
        """Initialize hasher.

        Args:
            hash_size: Size of hash (hash_size^2 bits)
        """
        self.hash_size = hash_size

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        """Load and convert image to grayscale."""
        try:
            img = Image.open(path).convert("L")
            return img
        except Exception as e:
            logger.warning(f"Cannot load image {path}: {e}")
            return None

    def dhash(self, path: Path) -> Optional[int]:
        """Compute difference hash.

        Compares adjacent pixels horizontally.

        Args:
            path: Path to image

        Returns:
            Hash as integer or None
        """
        img = self._load_image(path)
        if img is None:
            return None

        # Resize to hash_size+1 x hash_size
        img = img.resize((self.hash_size + 1, self.hash_size), Image.LANCZOS)
        pixels = np.array(img)

        # Compare adjacent pixels
        diff = pixels[:, 1:] > pixels[:, :-1]
        return int(np.packbits(diff.flatten()).tobytes().hex(), 16)

    def phash(self, path: Path) -> Optional[int]:
        """Compute perceptual hash using DCT.

        Args:
            path: Path to image

        Returns:
            Hash as integer or None
        """
        try:
            import imagehash
            img = Image.open(path)
            h = imagehash.phash(img, hash_size=self.hash_size)
            return int(str(h), 16)
        except ImportError:
            # Fallback to simple average hash
            return self.ahash(path)
        except Exception as e:
            logger.warning(f"Cannot compute phash for {path}: {e}")
            return None

    def ahash(self, path: Path) -> Optional[int]:
        """Compute average hash.

        Args:
            path: Path to image

        Returns:
            Hash as integer or None
        """
        img = self._load_image(path)
        if img is None:
            return None

        img = img.resize((self.hash_size, self.hash_size), Image.LANCZOS)
        pixels = np.array(img)
        avg = pixels.mean()
        diff = pixels > avg
        return int(np.packbits(diff.flatten()).tobytes().hex(), 16)

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        return bin(xor).count("1")


class EmbeddingComparator:
    """Compare images using CLIP embeddings."""

    def __init__(self, model_name: str = "ViT-B/32") -> None:
        """Initialize with CLIP model.

        Args:
            model_name: CLIP model name
        """
        self.model_name = model_name
        self._model = None
        self._preprocess = None

    def _load_model(self) -> None:
        """Load CLIP model lazily."""
        if self._model is not None:
            return

        try:
            import clip
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip.load(self.model_name, device=device)
            self._device = device
        except ImportError:
            logger.warning("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            raise

    def get_embedding(self, path: Path) -> Optional[np.ndarray]:
        """Get CLIP embedding for an image.

        Args:
            path: Path to image

        Returns:
            Embedding vector or None
        """
        self._load_model()

        try:
            import torch
            from PIL import Image

            img = Image.open(path).convert("RGB")
            img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                embedding = self._model.encode_image(img_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"Cannot get embedding for {path}: {e}")
            return None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2))


class DuplicateDetector:
    """Find duplicate and near-duplicate images.

    Supports multiple detection methods:
    - Perceptual hashing (fast, good for exact/near duplicates)
    - CLIP embeddings (slower, better for semantic similarity)

    Example:
        detector = DuplicateDetector(method="phash")
        result = detector.find_duplicates(image_paths, threshold=0.9)
        print(f"Found {result.num_duplicates} duplicates")
    """

    def __init__(
        self,
        method: str = "phash",
        hash_size: int = 8,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """Initialize detector.

        Args:
            method: Detection method (phash, dhash, ahash, clip)
            hash_size: Size for perceptual hashes
            progress_callback: Progress callback(current, total, message)
        """
        self.method = method
        self.hash_size = hash_size
        self.progress_callback = progress_callback

        if method in ("phash", "dhash", "ahash"):
            self._hasher = PerceptualHasher(hash_size)
        elif method == "clip":
            self._embedder = EmbeddingComparator()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress if callback set."""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _compute_hashes(
        self,
        paths: Sequence[Path],
    ) -> dict[Path, int]:
        """Compute hashes for all images.

        Args:
            paths: Image paths

        Returns:
            Dict of path -> hash
        """
        hash_func = getattr(self._hasher, self.method)
        hashes = {}
        total = len(paths)

        for idx, path in enumerate(paths):
            h = hash_func(path)
            if h is not None:
                hashes[path] = h
            self._report_progress(idx + 1, total, f"Hashing ({self.method})")

        return hashes

    def _compute_embeddings(
        self,
        paths: Sequence[Path],
    ) -> dict[Path, np.ndarray]:
        """Compute CLIP embeddings for all images.

        Args:
            paths: Image paths

        Returns:
            Dict of path -> embedding
        """
        embeddings = {}
        total = len(paths)

        for idx, path in enumerate(paths):
            emb = self._embedder.get_embedding(path)
            if emb is not None:
                embeddings[path] = emb
            self._report_progress(idx + 1, total, "Computing CLIP embeddings")

        return embeddings

    def find_duplicates(
        self,
        paths: Sequence[Path],
        threshold: float = 0.9,
    ) -> DuplicateResult:
        """Find duplicate images.

        Args:
            paths: List of image paths to check
            threshold: Similarity threshold (0-1, higher = more similar)

        Returns:
            DuplicateResult with groups of duplicates
        """
        paths = list(paths)

        if self.method == "clip":
            return self._find_duplicates_embedding(paths, threshold)
        else:
            return self._find_duplicates_hash(paths, threshold)

    def _find_duplicates_hash(
        self,
        paths: list[Path],
        threshold: float,
    ) -> DuplicateResult:
        """Find duplicates using perceptual hashing."""
        hashes = self._compute_hashes(paths)

        # Convert threshold to max Hamming distance
        max_bits = self.hash_size * self.hash_size
        max_distance = int((1 - threshold) * max_bits)

        # Find similar pairs
        groups: dict[Path, list[tuple[Path, float]]] = defaultdict(list)
        items = list(hashes.items())
        total_comparisons = len(items) * (len(items) - 1) // 2
        comparison = 0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                path1, hash1 = items[i]
                path2, hash2 = items[j]

                distance = self._hasher.hamming_distance(hash1, hash2)
                if distance <= max_distance:
                    similarity = 1 - (distance / max_bits)
                    groups[path1].append((path2, similarity))
                    groups[path2].append((path1, similarity))

                comparison += 1
                if comparison % 1000 == 0:
                    self._report_progress(comparison, total_comparisons, "Comparing hashes")

        # Cluster into groups using Union-Find
        return self._cluster_groups(paths, groups, threshold, self.method)

    def _find_duplicates_embedding(
        self,
        paths: list[Path],
        threshold: float,
    ) -> DuplicateResult:
        """Find duplicates using CLIP embeddings."""
        embeddings = self._compute_embeddings(paths)

        groups: dict[Path, list[tuple[Path, float]]] = defaultdict(list)
        items = list(embeddings.items())

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                path1, emb1 = items[i]
                path2, emb2 = items[j]

                similarity = self._embedder.cosine_similarity(emb1, emb2)
                if similarity >= threshold:
                    groups[path1].append((path2, similarity))
                    groups[path2].append((path1, similarity))

        return self._cluster_groups(paths, groups, threshold, "clip")

    def _cluster_groups(
        self,
        paths: list[Path],
        adjacency: dict[Path, list[tuple[Path, float]]],
        threshold: float,
        hash_type: str,
    ) -> DuplicateResult:
        """Cluster similar images into groups using Union-Find."""
        # Union-Find
        parent: dict[Path, Path] = {p: p for p in paths}

        def find(x: Path) -> Path:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: Path, y: Path) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union similar images
        for path, neighbors in adjacency.items():
            for neighbor, _ in neighbors:
                union(path, neighbor)

        # Group by root
        clusters: dict[Path, list[Path]] = defaultdict(list)
        for path in paths:
            if path in parent:  # Only if processed
                root = find(path)
                clusters[root].append(path)

        # Build groups (only groups with >1 image)
        groups = []
        for root, members in clusters.items():
            if len(members) > 1:
                # Collect similarity scores
                scores = []
                for path in members:
                    for neighbor, score in adjacency.get(path, []):
                        if neighbor in members:
                            scores.append(score)

                groups.append(DuplicateGroup(
                    images=sorted(members),
                    similarity_scores=scores,
                ))

        return DuplicateResult(
            groups=groups,
            total_images=len(paths),
            hash_type=hash_type,
            threshold=threshold,
        )


# Convenience function
def find_duplicates(
    paths: Sequence[Path],
    method: str = "phash",
    threshold: float = 0.9,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> DuplicateResult:
    """Find duplicate images.

    Args:
        paths: Image paths to check
        method: Detection method (phash, dhash, ahash, clip)
        threshold: Similarity threshold (0-1)
        progress_callback: Progress callback

    Returns:
        DuplicateResult with groups
    """
    detector = DuplicateDetector(method=method, progress_callback=progress_callback)
    return detector.find_duplicates(paths, threshold)
```

### 2. Create unit tests

Create `backend/tests/data/test_duplicates.py`:

```python
"""Tests for duplicate detection."""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from backend.data.duplicates import (
    DuplicateDetector,
    DuplicateResult,
    PerceptualHasher,
    find_duplicates,
)


@pytest.fixture
def sample_images(tmp_path):
    """Create sample images for testing."""
    # Create original image
    img1 = Image.new("RGB", (100, 100), color="red")
    img1.save(tmp_path / "img1.jpg")

    # Create exact duplicate
    shutil.copy(tmp_path / "img1.jpg", tmp_path / "img2.jpg")

    # Create near-duplicate (slightly modified)
    img3 = Image.new("RGB", (100, 100), color="red")
    img3.putpixel((50, 50), (255, 0, 0))  # Tiny change
    img3.save(tmp_path / "img3.jpg")

    # Create different image
    img4 = Image.new("RGB", (100, 100), color="blue")
    img4.save(tmp_path / "img4.jpg")

    return tmp_path


class TestPerceptualHasher:
    """Tests for PerceptualHasher."""

    def test_dhash(self, sample_images):
        """Test difference hash."""
        hasher = PerceptualHasher()
        h1 = hasher.dhash(sample_images / "img1.jpg")
        h2 = hasher.dhash(sample_images / "img2.jpg")

        assert h1 is not None
        assert h1 == h2  # Exact duplicates should have same hash

    def test_hamming_distance_identical(self, sample_images):
        """Test Hamming distance for identical images."""
        hasher = PerceptualHasher()
        h1 = hasher.dhash(sample_images / "img1.jpg")
        h2 = hasher.dhash(sample_images / "img2.jpg")

        distance = hasher.hamming_distance(h1, h2)
        assert distance == 0

    def test_hamming_distance_different(self, sample_images):
        """Test Hamming distance for different images."""
        hasher = PerceptualHasher()
        h1 = hasher.dhash(sample_images / "img1.jpg")
        h4 = hasher.dhash(sample_images / "img4.jpg")

        distance = hasher.hamming_distance(h1, h4)
        assert distance > 0


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""

    def test_find_exact_duplicates(self, sample_images):
        """Test finding exact duplicates."""
        paths = list(sample_images.glob("*.jpg"))
        detector = DuplicateDetector(method="phash")
        result = detector.find_duplicates(paths, threshold=0.99)

        assert isinstance(result, DuplicateResult)
        assert result.num_groups >= 1
        # img1 and img2 should be in same group
        group_with_duplicates = [g for g in result.groups if g.count >= 2]
        assert len(group_with_duplicates) >= 1

    def test_find_near_duplicates(self, sample_images):
        """Test finding near-duplicates."""
        paths = list(sample_images.glob("*.jpg"))
        detector = DuplicateDetector(method="dhash")
        result = detector.find_duplicates(paths, threshold=0.8)

        # Should find img1, img2, img3 as similar
        assert result.num_groups >= 1

    def test_different_images_not_grouped(self, sample_images):
        """Test that different images are not grouped."""
        paths = [sample_images / "img1.jpg", sample_images / "img4.jpg"]
        detector = DuplicateDetector(method="phash")
        result = detector.find_duplicates(paths, threshold=0.9)

        # Should have no groups (images are different)
        assert result.num_groups == 0

    def test_get_duplicates_to_remove(self, sample_images):
        """Test getting list of duplicates to remove."""
        paths = list(sample_images.glob("*.jpg"))
        result = find_duplicates(paths, method="phash", threshold=0.99)

        to_remove = result.get_duplicates_to_remove()
        # Should have at least one duplicate to remove
        assert len(to_remove) >= 1

    def test_progress_callback(self, sample_images):
        """Test progress callback is called."""
        paths = list(sample_images.glob("*.jpg"))
        progress_calls = []

        def callback(current, total, msg):
            progress_calls.append((current, total, msg))

        detector = DuplicateDetector(method="dhash", progress_callback=callback)
        detector.find_duplicates(paths, threshold=0.9)

        assert len(progress_calls) > 0
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/duplicates.py` | Create | Duplicate detection implementation |
| `backend/data/__init__.py` | Modify | Export duplicates module |
| `backend/tests/data/test_duplicates.py` | Create | Unit tests |

## Verification

```bash
cd backend
pytest tests/data/test_duplicates.py -v

# Quick test
python -c "
from pathlib import Path
from backend.data.duplicates import find_duplicates

# Test with some images
paths = list(Path('.').glob('**/*.jpg'))[:10]
if paths:
    result = find_duplicates(paths, method='phash', threshold=0.9)
    print(f'Found {result.num_groups} duplicate groups')
"
```

## Notes

- pHash uses DCT, more robust to scaling/compression
- dHash compares adjacent pixels, fast and simple
- aHash uses average intensity, most basic
- CLIP embeddings find semantic similarity (e.g., same object different angle)
- Threshold 0.9+ for near-exact duplicates, 0.7-0.8 for similar images
- imagehash package required for proper pHash implementation
