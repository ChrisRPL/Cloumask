"""
Model download infrastructure for CV models.

This module provides utilities for downloading model weights from
various sources including HuggingFace Hub and Ultralytics.

Implements spec: 03-cv-models/00-infrastructure
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default models directory (can be overridden by CLOUMASK_MODELS_DIR or settings)
_DEFAULT_MODELS_DIR = "models"


class ModelSource(str, Enum):
    """Supported model sources."""

    ULTRALYTICS = "ultralytics"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class ModelRegistryEntry:
    """Model registry entry with metadata."""

    name: str
    source: ModelSource
    size_mb: int
    filename: str | None = None
    repo_id: str | None = None  # For HuggingFace
    revision: str | None = None  # For HuggingFace
    requires_auth: bool = False
    checksum: str | None = None  # SHA256 checksum
    extra: dict[str, Any] = field(default_factory=dict)


# Type for progress callback: (downloaded_mb, total_mb) -> None
DownloadProgressCallback = Callable[[int, int], None]


def get_models_dir() -> Path:
    """
    Get the models directory path.

    Priority order:
    1. CLOUMASK_MODELS_DIR environment variable
    2. settings.models_dir (if backend.api.config is available)
    3. Default "models" directory

    Returns:
        Path to models directory.
    """
    # Check environment variable first
    env_dir = os.getenv("CLOUMASK_MODELS_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # Try to get from settings (lazy import to avoid hard dependency)
    try:
        from backend.api.config import settings

        return Path(settings.models_dir).resolve()
    except ImportError:
        logger.debug("backend.api.config not available, using default models dir")
        return Path(_DEFAULT_MODELS_DIR).resolve()


# Model registry with known models and their metadata
MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "yolo11m": ModelRegistryEntry(
        name="yolo11m",
        source=ModelSource.ULTRALYTICS,
        filename="yolo11m.pt",
        size_mb=40,
    ),
    "yolo11n": ModelRegistryEntry(
        name="yolo11n",
        source=ModelSource.ULTRALYTICS,
        filename="yolo11n.pt",
        size_mb=6,
    ),
    "sam2-hiera-large": ModelRegistryEntry(
        name="sam2-hiera-large",
        source=ModelSource.HUGGINGFACE,
        repo_id="facebook/sam2-hiera-large",
        size_mb=900,
        requires_auth=False,
    ),
    "sam2-hiera-base": ModelRegistryEntry(
        name="sam2-hiera-base",
        source=ModelSource.HUGGINGFACE,
        repo_id="facebook/sam2-hiera-base-plus",
        size_mb=320,
        requires_auth=False,
    ),
    "scrfd-10g": ModelRegistryEntry(
        name="scrfd-10g",
        source=ModelSource.HUGGINGFACE,
        repo_id="yuvalkirstain/scrfd_10g_bnkps",
        filename="scrfd_10g_bnkps.onnx",
        size_mb=16,
    ),
    "yolo-world-l": ModelRegistryEntry(
        name="yolo-world-l",
        source=ModelSource.ULTRALYTICS,
        filename="yolov8l-worldv2.pt",
        size_mb=170,
    ),
    "rtdetr-l": ModelRegistryEntry(
        name="rtdetr-l",
        source=ModelSource.ULTRALYTICS,
        filename="rtdetr-l.pt",
        size_mb=150,
    ),
    "sam3": ModelRegistryEntry(
        name="sam3",
        source=ModelSource.ULTRALYTICS,
        filename="sam3.pt",
        size_mb=3500,
        requires_auth=True,  # Gated model - requires HF approval
        extra={"hf_model_id": "facebook/sam3-hiera-large"},
    ),
    "sam2": ModelRegistryEntry(
        name="sam2",
        source=ModelSource.ULTRALYTICS,
        filename="sam2.1_b.pt",
        size_mb=400,
    ),
    "mobilesam": ModelRegistryEntry(
        name="mobilesam",
        source=ModelSource.ULTRALYTICS,
        filename="mobile_sam.pt",
        size_mb=40,
    ),
}


def get_model_path(name: str) -> Path:
    """
    Get local path for a model.

    Args:
        name: Model name from registry.

    Returns:
        Path where model should be stored.

    Raises:
        ValueError: If model not in registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[name]
    models_dir = get_models_dir()

    if entry.filename:
        return models_dir / entry.filename
    return models_dir / name


def is_model_downloaded(name: str) -> bool:
    """
    Check if model is already downloaded.

    Args:
        name: Model name from registry.

    Returns:
        True if model exists locally, False otherwise.
    """
    try:
        path = get_model_path(name)
        return path.exists()
    except ValueError:
        return False


def verify_checksum(path: Path, expected: str) -> bool:
    """
    Verify file SHA256 checksum.

    Args:
        path: Path to file.
        expected: Expected SHA256 hex digest.

    Returns:
        True if checksum matches, False otherwise.
    """
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    return actual.lower() == expected.lower()


def download_model(
    name: str,
    progress_callback: DownloadProgressCallback | None = None,
    force: bool = False,
) -> Path:
    """
    Download a model if not already present.

    Args:
        name: Model name from registry.
        progress_callback: Optional callback with (downloaded_mb, total_mb).
        force: Re-download even if exists.

    Returns:
        Path to downloaded model.

    Raises:
        ValueError: If model not in registry.
        RuntimeError: If download fails.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[name]
    path = get_model_path(name)

    # Skip if already downloaded and not forcing
    if not force and path.exists():
        logger.info("Model %s already downloaded at %s", name, path)
        return path

    # Ensure models directory exists
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading model %s (%.1f MB)...", name, entry.size_mb)

    try:
        if entry.source == ModelSource.ULTRALYTICS:
            return _get_ultralytics_path(name, entry, progress_callback)
        elif entry.source == ModelSource.HUGGINGFACE:
            return _download_huggingface(name, entry, progress_callback)
        elif entry.source == ModelSource.LOCAL:
            if not path.exists():
                raise RuntimeError(f"Local model not found: {path}")
            return path
        else:
            raise ValueError(f"Unknown source: {entry.source}")
    except Exception as e:
        logger.error("Failed to download model %s: %s", name, e)
        raise RuntimeError(f"Failed to download model {name}: {e}") from e


def _download_huggingface(
    name: str,
    entry: ModelRegistryEntry,
    progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    """
    Download model from HuggingFace Hub.

    Args:
        name: Model name.
        entry: Registry entry with repo details.
        progress_callback: Optional progress callback.

    Returns:
        Path to downloaded model.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    if not entry.repo_id:
        raise ValueError(f"Model {name} missing repo_id for HuggingFace download")

    # Get token from environment if auth required (support both common env var names)
    token = None
    if entry.requires_auth:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    models_dir = get_models_dir()
    local_dir = models_dir / name

    logger.info("Downloading from HuggingFace: %s", entry.repo_id)

    if entry.filename:
        # Download single file
        downloaded_path = hf_hub_download(
            repo_id=entry.repo_id,
            filename=entry.filename,
            revision=entry.revision,
            token=token,
            local_dir=models_dir,
        )
        return Path(downloaded_path)
    else:
        # Download entire repo
        snapshot_download(
            repo_id=entry.repo_id,
            revision=entry.revision,
            local_dir=local_dir,
            token=token,
        )
        return local_dir


def _get_ultralytics_path(
    name: str,
    entry: ModelRegistryEntry,
    progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    """
    Get expected path for Ultralytics model.

    Note: Ultralytics models auto-download on first use via their API.
    This function does NOT download the model - it only returns the
    expected path where the model will be stored. The actual download
    happens when the model is first loaded via ultralytics.YOLO().

    Args:
        name: Model name.
        entry: Registry entry.
        progress_callback: Optional progress callback (not used - Ultralytics
            handles download internally).

    Returns:
        Expected path where model will be stored.
    """
    models_dir = get_models_dir()
    model_path = models_dir / entry.filename if entry.filename else models_dir / f"{name}.pt"

    # Ensure directory exists for when Ultralytics downloads the model
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Ultralytics model %s will auto-download on first use to %s",
        name,
        model_path,
    )

    return model_path


def list_available_models() -> list[dict[str, Any]]:
    """
    List all models in registry with download status.

    Returns:
        List of dicts with model info and download status.
    """
    result = []
    for name, entry in MODEL_REGISTRY.items():
        downloaded = is_model_downloaded(name)
        result.append(
            {
                "name": name,
                "source": entry.source.value,
                "size_mb": entry.size_mb,
                "downloaded": downloaded,
                "requires_auth": entry.requires_auth,
                "path": str(get_model_path(name)) if downloaded else None,
            }
        )
    return result


def list_downloaded_models() -> list[str]:
    """
    List names of models that are downloaded.

    Returns:
        List of downloaded model names.
    """
    return [name for name in MODEL_REGISTRY if is_model_downloaded(name)]


def get_model_size_mb(name: str) -> int:
    """
    Get expected model size in MB.

    Args:
        name: Model name.

    Returns:
        Size in MB.

    Raises:
        ValueError: If model not in registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name].size_mb


def register_model(
    name: str,
    source: ModelSource | str,
    size_mb: int,
    *,
    filename: str | None = None,
    repo_id: str | None = None,
    requires_auth: bool = False,
    checksum: str | None = None,
) -> None:
    """
    Register a custom model in the registry.

    Args:
        name: Unique model name.
        source: Model source (ultralytics, huggingface, local).
        size_mb: Approximate size in MB.
        filename: Local filename (optional).
        repo_id: HuggingFace repo ID (for HF source).
        requires_auth: Whether HF auth is required.
        checksum: Optional SHA256 checksum.
    """
    if isinstance(source, str):
        source = ModelSource(source)

    MODEL_REGISTRY[name] = ModelRegistryEntry(
        name=name,
        source=source,
        size_mb=size_mb,
        filename=filename,
        repo_id=repo_id,
        requires_auth=requires_auth,
        checksum=checksum,
    )
    logger.info("Registered model: %s (source=%s, size=%dMB)", name, source.value, size_mb)


def delete_model(name: str) -> bool:
    """
    Delete a downloaded model.

    Args:
        name: Model name.

    Returns:
        True if deleted, False if not found.
    """
    if not is_model_downloaded(name):
        return False

    path = get_model_path(name)

    try:
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
        else:
            path.unlink()
        logger.info("Deleted model: %s", name)
        return True
    except Exception as e:
        logger.error("Failed to delete model %s: %s", name, e)
        return False


def get_total_downloaded_size_mb() -> int:
    """
    Get total size of all downloaded models in MB.

    Returns:
        Total size in MB.
    """
    return sum(entry.size_mb for name, entry in MODEL_REGISTRY.items() if is_model_downloaded(name))
