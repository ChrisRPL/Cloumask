"""
Understand node for parsing user requests and extracting intent.

This node uses the LLM to analyze natural language requests and
extract structured information about what the user wants to do.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm import SimpleLLMConfig, get_llm
from backend.agent.prompts import load_prompt
from backend.agent.state import MessageRole, PipelineState
from backend.agent.utils import extract_json_object
from backend.api.config import settings

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 2
FAST_LLM_TIMEOUT_SECONDS = 45
FAST_LLM_CONTEXT_TOKENS = 4096

PATH_PATTERN = re.compile(r"(/[\w\-.~/]+|~[/\w\-.]+|[A-Za-z]:\\[^\s]+)")
TRAILING_PATH_PUNCTUATION = ".,;:!?)]}\"'"
TRAILING_PATH_STOPWORDS = {
    "and",
    "then",
    "with",
    "to",
    "from",
    "for",
    "in",
    "on",
    "using",
    "please",
    "folder",
    "directory",
    "dataset",
    "project",
    "file",
    "files",
    "create",
    "detect",
    "detecting",
    "segment",
    "segmentation",
    "anonymize",
    "anonymise",
    "blur",
    "export",
    "convert",
    "save",
    "label",
    "annotate",
    "classify",
    "review",
    "execute",
    "run",
    "use",
    "model",
    "only",
    "results",
    # Class names commonly mentioned after paths
    "person",
    "people",
    "pedestrian",
    "pedestrians",
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "truck",
    "trucks",
    "bus",
    "bicycle",
    "bicycles",
    "bike",
    "bikes",
    "motorcycle",
    "motorcycles",
    "traffic",
    "light",
    "lights",
    "sign",
    "signs",
    "road",
}
QUOTED_PATH_PATTERN = re.compile(r"['\"]((?:/|~|[A-Za-z]:\\)[^'\"]+)['\"]")


def _sanitize_extracted_path(path_value: str) -> str:
    """Trim prose punctuation that may trail paths in natural language prompts."""
    cleaned = path_value.strip()
    while cleaned and cleaned[-1] in TRAILING_PATH_PUNCTUATION:
        cleaned = cleaned[:-1]
    return cleaned


def _extract_path(content: str) -> str | None:
    quoted_match = QUOTED_PATH_PATTERN.search(content)
    if quoted_match:
        cleaned = _sanitize_extracted_path(quoted_match.group(1))
        return cleaned or None

    match = PATH_PATTERN.search(content)
    if not match:
        return None
    raw_candidate = match.group(0)
    candidate = _sanitize_extracted_path(raw_candidate)
    suffix = content[match.end() :]
    candidate_had_trailing_punctuation = raw_candidate != candidate
    if suffix.startswith(" ") and not candidate_had_trailing_punctuation:
        extra_tokens: list[str] = []
        for token_match in re.finditer(r"\s+([A-Za-z0-9._-]+)", suffix):
            token = token_match.group(1)
            if token.lower() in TRAILING_PATH_STOPWORDS:
                break
            extra_tokens.append(token)
            if len(extra_tokens) >= 3:
                break
        if extra_tokens:
            candidate = f"{candidate} {' '.join(extra_tokens)}"

    return candidate or None


def _extract_output_path(content: str) -> str | None:
    """Extract an output/destination path from natural language."""
    # Match patterns like "save to /path", "output to /path", "export to /path", "into /path"
    output_patterns = [
        re.compile(r"(?:save|output|export|write|store)\s+(?:to|in|into|at)\s+['\"]?(/[\w\-.~/]+|~[/\w\-.]+)"),
        re.compile(r"(?:->|→|>>)\s*['\"]?(/[\w\-.~/]+|~[/\w\-.]+)"),
        re.compile(r"(?:results?|output)\s+(?:in|to)\s+['\"]?(/[\w\-.~/]+|~[/\w\-.]+)"),
    ]
    for pattern in output_patterns:
        match = pattern.search(content.lower() if "→" not in content else content)
        if match:
            return _sanitize_extracted_path(match.group(1))
    return None


def _extract_operations(content: str) -> list[str]:
    normalized = content.lower()
    # Track first occurrence position to preserve user-requested order.
    operation_hits: list[tuple[int, str]] = []
    # NOTE: Order matters — more specific multi-word aliases MUST come before
    # single-word ones to prevent premature matching. Longer alias lists are
    # checked first within each operation.
    operation_patterns = [
        ("scan", ["scan", "inspect", "list files", "show files", "what's in"]),
        ("detect", [
            "detect", "find objects", "object detection", "find cars",
            "find people", "find pedestrians", "find vehicles", "identify",
            "locate", "recognize",
        ]),
        ("segment", ["segment", "segmentation", "mask", "instance segmentation"]),
        ("anonymize", ["anonymize", "blur faces", "blur plates", "blur", "redact", "privacy", "gdpr"]),
        ("export", ["export", "save as", "export as", "save annotations"]),
        ("convert_format", [
            "convert to", "convert from", "convert format",
            "change format", "transform to", "reformat",
        ]),
        ("label", [
            "label", "auto-label", "autolabel", "auto label",
            "annotate", "annotation", "classify", "classification",
        ]),
        ("split", [
            "split", "split dataset", "train test", "train-test split",
            "train/val/test", "train val test", "create splits",
        ]),
        ("find_duplicates", [
            "find duplicates", "duplicates", "deduplicate", "dedup",
            "remove duplicates", "similar images", "near-duplicate",
            "duplicate detection",
        ]),
        ("label_qa", [
            "label qa", "validate labels", "check labels", "check annotations",
            "quality report", "annotation quality", "verify annotations",
            "validate annotations",
        ]),
        ("train", [
            "train a model", "train model", "train yolo", "training yolo",
            "train yolov", "training yolov", "yolov8 training", "yolov8 model",
            "fine-tune", "finetune", "fine tune",
            "train on this", "add step for training", "step for training",
        ]),
        ("script", [
            "run script", "execute script", "custom script", "run command",
            # Common non-standard operations that need a generated script:
            "augment", "data augmentation", "augmentation",
            "evaluate", "evaluation", "benchmark", "test model",
            "resize images", "resize", "rescale", "crop images",
            "visualize", "visualization", "generate report",
            "merge datasets", "merge", "combine datasets",
            "filter images", "filter dataset",
            "balance dataset", "class balancing", "oversample", "undersample",
            "compute statistics", "dataset statistics", "dataset stats",
        ]),
        ("review", ["review", "manual review", "human review", "send to review"]),
    ]

    for operation, aliases in operation_patterns:
        for alias in aliases:
            # Match complete terms only so "mask" does not match "Cloumask".
            match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized)
            if match:
                operation_hits.append((match.start(), operation))
                break

    operation_hits.sort(key=lambda hit: hit[0])
    operations: list[str] = []
    for _, operation in operation_hits:
        if operation not in operations:
            operations.append(operation)

    # Catch-all: "add step for X" / "include step for X" / "step for X"
    # where X is something not already matched.
    step_request = re.search(
        r"(?:add|include|insert|with|plus)\s+(?:a\s+)?(?:step|stage)\s+(?:for\s+)?(.+?)(?:\.|,|$)",
        normalized,
    )
    if step_request and "script" not in operations and "train" not in operations:
        operations.append("script")

    # Labeling implies detect + export workflow.
    if "label" in operations:
        if "detect" not in operations:
            operations.append("detect")
        if "export" not in operations:
            operations.append("export")

    # Infer composite workflows from context clues.
    operations = _infer_composite_operations(content, operations)

    return operations


def _infer_composite_operations(content: str, operations: list[str]) -> list[str]:
    """Expand operations based on implied multi-step workflows.

    Users often describe *goals* ("prepare dataset for training") rather than
    individual steps. This function detects those high-level intents and adds
    the missing intermediate operations so the agent can plan a complete
    pipeline in one shot.
    """
    normalized = content.lower()
    ops = list(operations)  # work on a copy

    # "prepare for training" / "train a model" / "training pipeline" → need
    # detect (if no annotations yet), export, and split.
    training_cues = [
        "prepare for training", "training pipeline", "train a model",
        "prepare dataset", "ready for training", "training data",
        "fine-tune", "finetune", "fine tune",
    ]
    wants_training = "train" in ops or any(cue in normalized for cue in training_cues)
    if wants_training or ("training" in normalized and "split" not in ops):
        if "detect" not in ops and "segment" not in ops:
            ops.append("detect")
        if "export" not in ops:
            ops.append("export")
        if "split" not in ops:
            ops.append("split")
        # Training is a custom workflow — ensure a script step is generated.
        if "train" in ops and "script" not in ops:
            ops.append("script")

    # "clean dataset" / "clean up" → find_duplicates
    clean_cues = ["clean dataset", "clean up", "cleanup", "remove bad"]
    if any(cue in normalized for cue in clean_cues):
        if "find_duplicates" not in ops:
            ops.append("find_duplicates")

    # "review" without other ops should still detect first if there are no
    # existing annotations to review.
    if "review" in ops and len(ops) == 1:
        ops.insert(0, "detect")

    return ops


def _extract_parameters(content: str) -> dict[str, Any]:
    normalized = content.lower()
    parameters: dict[str, Any] = {}

    # Common detection classes for quick heuristic extraction.
    class_terms = {
        "person": ["person", "people", "pedestrian", "pedestrians", "pedestrains"],
        "car": ["car", "cars", "vehicle", "vehicles"],
        "truck": ["truck", "trucks"],
        "bus": ["bus", "buses"],
        "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
        "traffic light": ["traffic light", "traffic lights"],
        "road sign": ["road sign", "road signs", "sign", "signs"],
    }
    classes: list[str] = []
    for canonical, aliases in class_terms.items():
        if any(alias in normalized for alias in aliases):
            classes.append(canonical)
    if classes:
        parameters["classes"] = classes

    confidence_match = re.search(r"confidence\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", normalized)
    if confidence_match:
        parameters["confidence"] = float(confidence_match.group(1))

    format_match = re.search(
        r"\b(yolo|coco|kitti|voc|pascal|cvat|nuscenes|openlabel)\b",
        normalized,
    )
    if format_match:
        fmt = format_match.group(1)
        parameters["format"] = "voc" if fmt == "pascal" else fmt

    # Detect training requests — prefer specific model names over bare "model".
    training_match = re.search(r"\btrain(?:ing)?\b.*\b(yolov?\d+)\b", normalized)
    if not training_match:
        training_match = re.search(r"\btrain(?:ing)?\b.*\b(ultralytics|model)\b", normalized)
    if training_match:
        parameters["training"] = True
        model_name = training_match.group(1)
        parameters["model_type"] = model_name if model_name != "model" else "yolov8"

    # Extract custom step description from "add step for X" patterns.
    step_desc = re.search(
        r"(?:add|include|insert|with|plus)\s+(?:a\s+)?(?:step|stage)\s+(?:for\s+)?(.+?)(?:\.|,|$)",
        normalized,
    )
    if step_desc:
        parameters["custom_step_description"] = step_desc.group(1).strip()

    face_mentioned = bool(re.search(r"\bfaces?\b", normalized))
    plate_mentioned = bool(re.search(r"\b(?:license\s+)?plates?\b", normalized))

    faces_only = bool(re.search(r"\b(?:only|just)\s+faces?\b", normalized)) or bool(
        re.search(
            r"\b(?:no|not|without|exclude|excluding)\s+(?:anonymize\s+)?(?:license\s+)?plates?\b",
            normalized,
        )
    )
    plates_only = bool(
        re.search(r"\b(?:only|just)\s+(?:license\s+)?plates?\b", normalized)
    ) or bool(
        re.search(
            r"\b(?:no|not|without|exclude|excluding)\s+(?:anonymize\s+)?faces?\b",
            normalized,
        )
    )

    if face_mentioned or plate_mentioned:
        if faces_only and not plates_only:
            parameters["target"] = "faces"
        elif plates_only and not faces_only:
            parameters["target"] = "plates"
        elif face_mentioned and plate_mentioned:
            parameters["target"] = "all"
        elif face_mentioned:
            parameters["target"] = "faces"
        elif plate_mentioned:
            parameters["target"] = "plates"

    return parameters


def _infer_input_type(content: str, input_path: str | None) -> str:
    normalized = content.lower()
    if input_path:
        lower_path = input_path.lower()
        if any(ext in lower_path for ext in (".pcd", ".ply", ".las", ".laz", ".bin")):
            return "pointcloud"
        if any(ext in lower_path for ext in (".mp4", ".mov", ".avi", ".mkv")):
            return "video"
    if "point cloud" in normalized or "lidar" in normalized:
        return "pointcloud"
    if "video" in normalized:
        return "video"
    return "images"


def _build_fast_understanding(content: str) -> dict[str, Any] | None:
    """
    Build understanding without LLM for clear multi-operation task requests.

    We deliberately avoid fast-path for pure "scan" requests to keep broad
    compatibility with existing planning tests and behavior.
    """
    input_path = _extract_path(content)
    operations = _extract_operations(content)
    if not input_path or not operations:
        return None

    if len(operations) == 1 and operations[0] == "scan":
        return None

    parameters = _extract_parameters(content)
    output_path = _extract_output_path(content)

    return {
        "intent": operations[0],
        "input_path": input_path,
        "input_type": _infer_input_type(content, input_path),
        "operations": operations,
        "parameters": parameters,
        "output_path": output_path,
        "clarification_needed": None,
    }


async def understand(state: PipelineState) -> dict[str, Any]:
    """
    Analyze user request and extract structured intent.

    This node:
    1. Gets the latest user message
    2. Sends it to the LLM with the understand prompt
    3. Parses the JSON response to extract intent
    4. Stores understanding in metadata or asks for clarification

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with understanding in metadata or clarification message.
    """
    # Get the latest user message
    messages = state.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == MessageRole.USER.value]

    if not user_messages:
        logger.warning("No user message found in state")
        return {"last_error": "No user message found"}

    latest_user_msg = user_messages[-1]["content"]
    logger.info(f"Understanding request: {latest_user_msg[:100]}...")

    # Fast-path deterministic understanding for clear task requests.
    fast_understanding = _build_fast_understanding(latest_user_msg)
    if fast_understanding is not None:
        metadata = {**state.get("metadata", {}), "understanding": fast_understanding}
        operations = fast_understanding.get("operations", [])
        input_path = fast_understanding.get("input_path") or "your data"
        operation_text = ", ".join(operations) if operations else "process"

        new_message = {
            "role": MessageRole.ASSISTANT.value,
            "content": (
                f"I understand you want to {operation_text} on {input_path}. Let me create a plan."
            ),
            "timestamp": datetime.now().isoformat(),
            "tool_calls": [],
            "tool_call_id": None,
        }

        return {
            "messages": [*messages, new_message],
            "metadata": metadata,
        }

    # Load the understand prompt
    try:
        understand_prompt = load_prompt("understand")
    except FileNotFoundError:
        logger.error("Failed to load understand prompt")
        return {"last_error": "Failed to load understand prompt"}

    # Build LLM messages
    llm_messages = [
        SystemMessage(content=understand_prompt),
        HumanMessage(content=latest_user_msg),
    ]

    # Get LLM and call with retries
    llm = get_llm(
        config=SimpleLLMConfig(
            host=settings.ollama_host,
            model=settings.ollama_model,
            temperature=0.1,
            timeout=FAST_LLM_TIMEOUT_SECONDS,
            num_ctx=FAST_LLM_CONTEXT_TOKENS,
        ),
        temperature=0.1,  # Low temperature for structured output
    )

    last_error: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.ainvoke(llm_messages)
            response_content = response.content

            # Handle string or list content
            if isinstance(response_content, list):
                response_content = " ".join(str(item) for item in response_content if item)

            logger.debug(f"LLM response (attempt {attempt + 1}): {response_content}")

            # Parse JSON from response
            understanding = extract_json_object(str(response_content))

            if understanding is None:
                last_error = f"Failed to parse JSON from response: {response_content[:200]}"
                logger.warning(last_error)
                continue

            # Check if clarification is needed
            clarification = understanding.get("clarification_needed")
            if clarification:
                logger.info(f"Clarification needed: {clarification}")
                new_message = {
                    "role": MessageRole.ASSISTANT.value,
                    "content": clarification,
                    "timestamp": datetime.now().isoformat(),
                    "tool_calls": [],
                    "tool_call_id": None,
                }
                return {
                    "messages": [*messages, new_message],
                    "awaiting_user": True,
                }

            # Store understanding in metadata (avoid mutating original state)
            metadata = {**state.get("metadata", {}), "understanding": understanding}
            logger.info(f"Understood intent: {understanding.get('intent')}")

            # Build confirmation message
            operations = understanding.get("operations", [])
            if not operations and understanding.get("intent"):
                operations = [understanding["intent"]]

            input_path = understanding.get("input_path") or "your data"
            operation_text = ", ".join(operations) if operations else "process"

            confirmation = (
                f"I understand you want to {operation_text} on {input_path}. Let me create a plan."
            )

            new_message = {
                "role": MessageRole.ASSISTANT.value,
                "content": confirmation,
                "timestamp": datetime.now().isoformat(),
                "tool_calls": [],
                "tool_call_id": None,
            }

            return {
                "messages": [*messages, new_message],
                "metadata": metadata,
            }

        except Exception as e:
            last_error = f"LLM call failed: {e}"
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                break

    # All retries failed
    logger.error(f"All {MAX_RETRIES} attempts failed: {last_error}")
    error_message: dict[str, Any] = {
        "role": MessageRole.ASSISTANT.value,
        "content": "I'm having trouble understanding your request. Could you please rephrase it?",
        "timestamp": datetime.now().isoformat(),
        "tool_calls": [],
        "tool_call_id": None,
    }

    return {
        "messages": [*messages, error_message],
        "last_error": last_error,
        "awaiting_user": True,
    }


__all__ = ["understand"]
