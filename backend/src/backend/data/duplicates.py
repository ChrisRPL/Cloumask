"""Duplicate and near-duplicate image detection utilities.

Supports perceptual hashing (pHash, dHash, aHash) and optional CLIP
embeddings for semantic similarity.

Implements spec: 06-data-pipeline/17-duplicate-detection
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class DuplicateGroup:
    """A group of duplicate/similar images."""

    images: list[Path]
    similarity_scores: list[float] = field(default_factory=list)

    @property
    def representative(self) -> Path:
        """Return first image as group representative."""
        return self.images[0] if self.images else Path()

    @property
    def duplicates(self) -> list[Path]:
        """Return all images except representative."""
        return self.images[1:] if len(self.images) > 1 else []

    @property
    def count(self) -> int:
        """Return total number of images in the group."""
        return len(self.images)


@dataclass
class DuplicateResult:
    """Result object for duplicate detection."""

    groups: list[DuplicateGroup]
    total_images: int
    method: str
    threshold: float

    @property
    def num_duplicates(self) -> int:
        """Total duplicate images excluding group representatives."""
        return sum(group.count - 1 for group in self.groups)

    @property
    def num_groups(self) -> int:
        """Number of duplicate groups."""
        return len(self.groups)

    def get_duplicates_to_remove(self) -> list[Path]:
        """Return duplicate paths to remove while keeping representatives."""
        return [path for group in self.groups for path in group.duplicates]


class PerceptualHasher:
    """Compute perceptual hashes for image similarity."""

    _dct_cache: dict[int, NDArray[np.float64]] = {}

    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4) -> None:
        if hash_size < 2:
            raise ValueError("hash_size must be >= 2")
        if highfreq_factor < 2:
            raise ValueError("highfreq_factor must be >= 2")

        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

    def _load_grayscale(self, path: Path) -> Image.Image | None:
        """Load image as grayscale (L mode)."""
        try:
            with Image.open(path) as img:
                return cast(Image.Image, img.convert("L"))
        except OSError as exc:
            logger.warning("Cannot load image %s: %s", path, exc)
            return None

    @staticmethod
    def _pack_bits(bits: NDArray[np.bool_]) -> int:
        """Pack boolean hash bits into an integer."""
        flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
        padding = (-flat.size) % 8
        if padding:
            flat = np.pad(flat, (0, padding), constant_values=0).reshape(-1)
        packed = np.packbits(flat)
        return int.from_bytes(packed.tobytes(), byteorder="big", signed=False)

    @classmethod
    def _dct_matrix(cls, size: int) -> NDArray[np.float64]:
        """Create (and cache) orthonormal DCT-II transform matrix."""
        cached = cls._dct_cache.get(size)
        if cached is not None:
            return cached

        matrix = np.zeros((size, size), dtype=np.float64)
        factor = math.pi / (2.0 * size)
        scale0 = math.sqrt(1.0 / size)
        scale = math.sqrt(2.0 / size)

        for k in range(size):
            alpha = scale0 if k == 0 else scale
            for n in range(size):
                matrix[k, n] = alpha * math.cos((2 * n + 1) * k * factor)

        cls._dct_cache[size] = matrix
        return matrix

    def ahash(self, path: Path) -> int | None:
        """Average hash using grayscale mean thresholding."""
        img = self._load_grayscale(path)
        if img is None:
            return None

        resized = img.resize((self.hash_size, self.hash_size), resample=Image.Resampling.LANCZOS)
        pixels = np.asarray(resized, dtype=np.float64)
        threshold = float(pixels.mean())
        bits = pixels > threshold
        return self._pack_bits(bits)

    def dhash(self, path: Path) -> int | None:
        """Difference hash using adjacent horizontal pixel differences."""
        img = self._load_grayscale(path)
        if img is None:
            return None

        resized = img.resize(
            (self.hash_size + 1, self.hash_size),
            resample=Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(resized, dtype=np.float64)
        bits = pixels[:, 1:] > pixels[:, :-1]
        return self._pack_bits(bits)

    def phash(self, path: Path) -> int | None:
        """Perceptual hash using low-frequency DCT coefficients."""
        img = self._load_grayscale(path)
        if img is None:
            return None

        dct_size = self.hash_size * self.highfreq_factor
        resized = img.resize((dct_size, dct_size), resample=Image.Resampling.LANCZOS)
        pixels = np.asarray(resized, dtype=np.float64)

        dct = self._dct_matrix(dct_size)
        dct_coeff = np.dot(np.dot(dct, pixels), dct.T)
        low_freq = dct_coeff[: self.hash_size, : self.hash_size]

        flat = low_freq.reshape(-1)
        median = float(np.median(flat[1:])) if flat.size > 1 else float(flat[0])
        bits = low_freq > median
        return self._pack_bits(bits)

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two integer hashes."""
        return (hash1 ^ hash2).bit_count()


class EmbeddingComparator:
    """Compare images with CLIP embeddings and cosine similarity."""

    def __init__(self, model_name: str = "ViT-B/32") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._preprocess: Callable[[Image.Image], Any] | None = None
        self._device = "cpu"

    def _load_model(self) -> None:
        """Load CLIP model lazily."""
        if self._model is not None and self._preprocess is not None:
            return

        try:
            import clip  # type: ignore[import-not-found]
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "CLIP dependencies are missing. Install with: "
                "pip install git+https://github.com/openai/CLIP.git torch"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(self.model_name, device=self._device)
        model.eval()
        self._model = model
        self._preprocess = preprocess

    def get_embedding(self, path: Path) -> NDArray[np.float32] | None:
        """Compute normalized CLIP embedding for one image."""
        self._load_model()

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for CLIP embedding inference") from exc

        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")

            assert self._preprocess is not None
            assert self._model is not None
            image_tensor = self._preprocess(img_rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                embedding = self._model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            array = cast(
                NDArray[np.float32],
                embedding.cpu().numpy().astype(np.float32).reshape(-1),
            )
            return array
        except OSError as exc:
            logger.warning("Cannot load image for embedding %s: %s", path, exc)
            return None
        except Exception as exc:
            logger.warning("Cannot compute embedding for %s: %s", path, exc)
            return None

    @staticmethod
    def cosine_similarity(emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """Compute cosine similarity for two vectors."""
        denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if denom <= 0:
            return 0.0
        return float(np.dot(emb1, emb2) / denom)


class DuplicateDetector:
    """Find duplicate and near-duplicate images using hashes or CLIP embeddings."""

    SUPPORTED_METHODS = {"phash", "dhash", "ahash", "clip"}

    def __init__(
        self,
        method: str = "phash",
        hash_size: int = 8,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported methods: {', '.join(sorted(self.SUPPORTED_METHODS))}"
            )
        self.method = method
        self.hash_size = hash_size
        self.progress_callback = progress_callback

        self._hasher: PerceptualHasher | None = None
        self._embedder: EmbeddingComparator | None = None
        if method == "clip":
            self._embedder = EmbeddingComparator()
        else:
            self._hasher = PerceptualHasher(hash_size=hash_size)

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    def _report_progress(self, current: int, total: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _compute_hashes(self, paths: Sequence[Path]) -> dict[Path, int]:
        """Compute perceptual hashes for images."""
        assert self._hasher is not None

        hash_fn: Callable[[Path], int | None]
        if self.method == "phash":
            hash_fn = self._hasher.phash
        elif self.method == "dhash":
            hash_fn = self._hasher.dhash
        else:
            hash_fn = self._hasher.ahash

        hashes: dict[Path, int] = {}
        total = len(paths)
        for idx, path in enumerate(paths, start=1):
            value = hash_fn(path)
            if value is not None:
                hashes[path] = value
            self._report_progress(idx, total, f"Hashing ({self.method})")
        return hashes

    def _compute_embeddings(self, paths: Sequence[Path]) -> dict[Path, NDArray[np.float32]]:
        """Compute CLIP embeddings for images."""
        assert self._embedder is not None

        embeddings: dict[Path, NDArray[np.float32]] = {}
        total = len(paths)
        for idx, path in enumerate(paths, start=1):
            emb = self._embedder.get_embedding(path)
            if emb is not None:
                embeddings[path] = emb
            self._report_progress(idx, total, "Computing CLIP embeddings")
        return embeddings

    def find_duplicates(self, paths: Sequence[Path], threshold: float = 0.9) -> DuplicateResult:
        """Find duplicate and near-duplicate image groups."""
        self._validate_threshold(threshold)

        unique_paths = [Path(path) for path in dict.fromkeys(paths)]
        if len(unique_paths) < 2:
            return DuplicateResult(
                groups=[],
                total_images=len(unique_paths),
                method=self.method,
                threshold=threshold,
            )

        if self.method == "clip":
            return self._find_duplicates_embedding(unique_paths, threshold)
        return self._find_duplicates_hash(unique_paths, threshold)

    def _find_duplicates_hash(self, paths: list[Path], threshold: float) -> DuplicateResult:
        """Find duplicates using perceptual hash similarity."""
        hashes = self._compute_hashes(paths)
        items = list(hashes.items())
        adjacency: dict[Path, list[tuple[Path, float]]] = defaultdict(list)

        total_comparisons = len(items) * (len(items) - 1) // 2
        done_comparisons = 0
        max_bits = self.hash_size * self.hash_size

        assert self._hasher is not None
        for i, (path1, hash1) in enumerate(items):
            for path2, hash2 in items[i + 1 :]:
                distance = self._hasher.hamming_distance(hash1, hash2)
                similarity = 1.0 - (distance / max_bits)
                if similarity >= threshold:
                    adjacency[path1].append((path2, similarity))
                    adjacency[path2].append((path1, similarity))

                done_comparisons += 1
                if total_comparisons and (
                    done_comparisons % 1000 == 0 or done_comparisons == total_comparisons
                ):
                    self._report_progress(done_comparisons, total_comparisons, "Comparing hashes")

        groups = self._cluster_groups(list(hashes.keys()), adjacency)
        return DuplicateResult(
            groups=groups,
            total_images=len(paths),
            method=self.method,
            threshold=threshold,
        )

    def _find_duplicates_embedding(self, paths: list[Path], threshold: float) -> DuplicateResult:
        """Find duplicates using CLIP cosine similarity."""
        embeddings = self._compute_embeddings(paths)
        items = list(embeddings.items())
        adjacency: dict[Path, list[tuple[Path, float]]] = defaultdict(list)

        total_comparisons = len(items) * (len(items) - 1) // 2
        done_comparisons = 0

        assert self._embedder is not None
        for i, (path1, emb1) in enumerate(items):
            for path2, emb2 in items[i + 1 :]:
                similarity = self._embedder.cosine_similarity(emb1, emb2)
                if similarity >= threshold:
                    adjacency[path1].append((path2, similarity))
                    adjacency[path2].append((path1, similarity))

                done_comparisons += 1
                if total_comparisons and (
                    done_comparisons % 500 == 0 or done_comparisons == total_comparisons
                ):
                    self._report_progress(
                        done_comparisons,
                        total_comparisons,
                        "Comparing CLIP embeddings",
                    )

        groups = self._cluster_groups(list(embeddings.keys()), adjacency)
        return DuplicateResult(
            groups=groups,
            total_images=len(paths),
            method="clip",
            threshold=threshold,
        )

    @staticmethod
    def _cluster_groups(
        paths: list[Path],
        adjacency: dict[Path, list[tuple[Path, float]]],
    ) -> list[DuplicateGroup]:
        """Cluster images into duplicate groups with union-find."""
        if not paths:
            return []

        parent: dict[Path, Path] = {path: path for path in paths}

        def find(path: Path) -> Path:
            root = path
            while parent[root] != root:
                root = parent[root]
            while path != root:
                next_path = parent[path]
                parent[path] = root
                path = next_path
            return root

        def union(path_a: Path, path_b: Path) -> None:
            root_a = find(path_a)
            root_b = find(path_b)
            if root_a != root_b:
                parent[root_a] = root_b

        for path, neighbors in adjacency.items():
            for neighbor, _ in neighbors:
                if path in parent and neighbor in parent:
                    union(path, neighbor)

        clusters: dict[Path, list[Path]] = defaultdict(list)
        for path in paths:
            clusters[find(path)].append(path)

        groups: list[DuplicateGroup] = []
        for members in clusters.values():
            if len(members) < 2:
                continue

            sorted_members = sorted(members)
            member_set = set(sorted_members)
            pair_scores: list[float] = []
            for index, path in enumerate(sorted_members):
                neighbor_scores = {
                    neighbor: score
                    for neighbor, score in adjacency.get(path, [])
                    if neighbor in member_set
                }
                for other in sorted_members[index + 1 :]:
                    if other in neighbor_scores:
                        pair_scores.append(neighbor_scores[other])

            groups.append(
                DuplicateGroup(
                    images=sorted_members,
                    similarity_scores=pair_scores,
                )
            )

        groups.sort(key=lambda group: (-group.count, str(group.representative)))
        return groups


def find_duplicates(
    paths: Sequence[Path],
    method: str = "phash",
    threshold: float = 0.9,
    hash_size: int = 8,
    progress_callback: ProgressCallback | None = None,
) -> DuplicateResult:
    """Convenience helper for duplicate detection."""
    detector = DuplicateDetector(
        method=method,
        hash_size=hash_size,
        progress_callback=progress_callback,
    )
    return detector.find_duplicates(paths, threshold=threshold)
