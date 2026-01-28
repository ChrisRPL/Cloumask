"""
Script behavior schema definitions.

Defines structured representations of what a script does without exposing the code.
Used by the BehaviorCard component to display script behavior to users.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BehaviorInput:
    """Describes an input that a script expects."""

    name: str
    """Human-readable name (e.g., "Images", "Point Cloud")."""

    types: list[str]
    """File extensions accepted (e.g., [".jpg", ".png"])."""

    description: str
    """What this input is used for."""

    required: bool = True
    """Whether this input is required."""


@dataclass
class BehaviorOutput:
    """Describes an output that a script produces."""

    name: str
    """Human-readable name (e.g., "Processed Images", "Annotations")."""

    types: list[str]
    """File extensions produced (e.g., [".jpg", ".json"])."""

    description: str
    """What this output contains."""


@dataclass
class ResourceUsage:
    """Describes resource requirements for a script."""

    cpu: Literal["low", "medium", "high"] = "low"
    """CPU intensity."""

    memory: Literal["low", "medium", "high"] = "low"
    """Memory usage."""

    gpu: bool = False
    """Whether GPU is used/required."""


@dataclass
class ScriptBehavior:
    """
    Structured representation of what a script does.

    This is shown to users instead of raw code, making scripts
    understandable without programming knowledge.
    """

    inputs: list[BehaviorInput] = field(default_factory=list)
    """Files/data the script receives."""

    outputs: list[BehaviorOutput] = field(default_factory=list)
    """Files/data the script produces."""

    operations: list[str] = field(default_factory=list)
    """What the script does (bullet list of actions)."""

    warnings: list[str] = field(default_factory=list)
    """Potential issues or edge cases."""

    estimated_time: str | None = None
    """Estimated processing time (e.g., "~2s per image")."""

    resource_usage: ResourceUsage | None = None
    """Resource requirements."""


# Common input/output presets for CV operations
PRESET_IMAGE_INPUT = BehaviorInput(
    name="Images",
    types=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    description="Input images to process",
)

PRESET_ANNOTATION_INPUT = BehaviorInput(
    name="Annotations",
    types=[".json", ".xml", ".txt"],
    description="Existing annotations in COCO, YOLO, or Pascal VOC format",
)

PRESET_IMAGE_OUTPUT = BehaviorOutput(
    name="Processed Images",
    types=[".jpg", ".png"],
    description="Processed/transformed images",
)

PRESET_ANNOTATION_OUTPUT = BehaviorOutput(
    name="Annotations",
    types=[".json"],
    description="Generated annotations in COCO format",
)

PRESET_MASK_OUTPUT = BehaviorOutput(
    name="Segmentation Masks",
    types=[".png"],
    description="Binary or multi-class segmentation masks",
)
