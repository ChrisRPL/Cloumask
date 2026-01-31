"""Thumbnail generation utilities for review queue."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


def generate_thumbnail(
    image_path: str | Path,
    max_size: int = 320,
    quality: int = 85
) -> str:
    """
    Generate a base64-encoded thumbnail for efficient list view rendering.

    Args:
        image_path: Path to the source image file
        max_size: Maximum dimension (width or height) in pixels
        quality: JPEG quality (0-100), default 85

    Returns:
        Base64-encoded data URL (data:image/jpeg;base64,...)

    Raises:
        FileNotFoundError: If image file doesn't exist
        PIL.UnidentifiedImageError: If file is not a valid image
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Open and resize image
    img = Image.open(path)

    # Convert RGBA to RGB if necessary
    if img.mode == "RGBA":
        # Create white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Encode to JPEG in memory
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)

    # Base64 encode
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_data}"


def get_image_dimensions(image_path: str | Path) -> tuple[int, int]:
    """
    Get image dimensions without loading the full image into memory.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        FileNotFoundError: If image file doesn't exist
        PIL.UnidentifiedImageError: If file is not a valid image
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(path) as img:
        return img.size  # (width, height)
