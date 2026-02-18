"""
Plan node for generating execution plans from understood requests.

This node uses the LLM to create a structured execution plan
based on the user's intent extracted by the understand node.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm import SimpleLLMConfig, get_llm
from backend.agent.prompts import load_prompt
from backend.agent.state import MessageRole, PipelineState, StepStatus
from backend.agent.tools.discovery import initialize_tools
from backend.agent.tools.registry import get_tool_registry
from backend.agent.utils import extract_json_array
from backend.api.config import settings

if TYPE_CHECKING:
    from backend.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 2
FAST_LLM_TIMEOUT_SECONDS = 45
FAST_LLM_CONTEXT_TOKENS = 4096

# Valid tool names that can appear in plans
VALID_TOOLS = frozenset(
    [
        "scan_directory",
        "anonymize",
        "detect",
        "segment",
        "export",
        "convert_format",
        "find_duplicates",
        "label_qa",
        "split_dataset",
        "run_script",
        "review",
        "pointcloud_stats",
        "process_pointcloud",
        "detect_3d",
        "project_3d_to_2d",
        "anonymize_pointcloud",
        "extract_rosbag",
    ]
)


def _get_registered_tools() -> dict[str, BaseTool]:
    """Return registered tools, initializing discovery if needed."""
    registry = get_tool_registry()
    if len(registry) == 0:
        initialize_tools()
    return {tool.name: tool for tool in registry.get_all()}


def _validate_core_step_parameters(
    step_num: int,
    tool_name: str,
    parameters: dict[str, Any],
) -> str | None:
    """
    Validate required parameters for core planner tools.

    This preserves backwards-compatible behavior for runtime environments where
    registry discovery may not expose every core tool.
    """
    if tool_name == "scan_directory":
        if "path" not in parameters:
            return f"Step {step_num} (scan_directory) missing required 'path' parameter"

    elif tool_name == "anonymize":
        if "input_path" not in parameters:
            return f"Step {step_num} (anonymize) missing required 'input_path' parameter"
        if "output_path" not in parameters:
            return f"Step {step_num} (anonymize) missing required 'output_path' parameter"

    elif tool_name == "detect":
        if "input_path" not in parameters:
            return f"Step {step_num} (detect) missing required 'input_path' parameter"

    elif tool_name == "segment":
        if "input_path" not in parameters:
            return f"Step {step_num} (segment) missing required 'input_path' parameter"
        if "prompt" not in parameters:
            return f"Step {step_num} (segment) missing required 'prompt' parameter"

    elif tool_name == "export":
        if "source_path" not in parameters:
            return f"Step {step_num} (export) missing required 'source_path' parameter"
        if "output_path" not in parameters:
            return f"Step {step_num} (export) missing required 'output_path' parameter"
        if "output_format" not in parameters:
            return f"Step {step_num} (export) missing required 'output_format' parameter"

    elif tool_name == "convert_format":
        if "source_path" not in parameters:
            return f"Step {step_num} (convert_format) missing required 'source_path' parameter"
        if "output_path" not in parameters:
            return f"Step {step_num} (convert_format) missing required 'output_path' parameter"
        if "target_format" not in parameters:
            return f"Step {step_num} (convert_format) missing required 'target_format' parameter"

    elif tool_name == "find_duplicates" and "path" not in parameters:
        return f"Step {step_num} (find_duplicates) missing required 'path' parameter"

    elif tool_name == "label_qa" and "path" not in parameters:
        return f"Step {step_num} (label_qa) missing required 'path' parameter"

    elif tool_name == "split_dataset":
        if "path" not in parameters:
            return f"Step {step_num} (split_dataset) missing required 'path' parameter"
        if "output_path" not in parameters:
            return f"Step {step_num} (split_dataset) missing required 'output_path' parameter"

    elif tool_name in {"run_script", "custom_script"}:
        if "input_path" not in parameters:
            return f"Step {step_num} ({tool_name}) missing required 'input_path' parameter"

    elif tool_name == "review":
        if "source_path" not in parameters:
            return f"Step {step_num} (review) missing required 'source_path' parameter"
        if "image_dir" not in parameters:
            return f"Step {step_num} (review) missing required 'image_dir' parameter"

    return None


def _normalize_classes(value: Any) -> list[str]:
    if isinstance(value, list):
        classes = [str(item).strip() for item in value if str(item).strip()]
        return classes
    if isinstance(value, str):
        classes = [part.strip() for part in value.split(",") if part.strip()]
        return classes
    return []


def _looks_like_inline_python_script(value: Any) -> bool:
    """Heuristic for distinguishing inline code from script file paths."""
    if not isinstance(value, str):
        return False

    content = value.strip()
    if not content:
        return False

    if "\n" in content:
        return True

    if content.lower().startswith("#!/usr/bin/env python"):
        return True

    python_markers = ("def process(", "import ", "from ", "class ", "return ")
    if any(marker in content for marker in python_markers):
        return True

    return False


def _normalize_run_script_step(step: dict[str, Any]) -> dict[str, Any]:
    """Move inline run_script code into step-level generated_code."""
    normalized = dict(step)
    if normalized.get("tool_name") != "run_script":
        return normalized

    parameters = dict(normalized.get("parameters", {}))
    generated_code = normalized.get("generated_code")
    if not isinstance(generated_code, str) or not generated_code.strip():
        script_value = parameters.get("script")
        if _looks_like_inline_python_script(script_value):
            generated_code = str(script_value).strip()
            parameters.pop("script", None)

    if isinstance(generated_code, str) and generated_code.strip():
        normalized["generated_code"] = generated_code.strip()

    normalized["parameters"] = parameters
    return normalized


def _to_pipeline_step(step: dict[str, Any], step_index: int) -> dict[str, Any]:
    """Normalize planner output into persisted pipeline step shape."""
    normalized = _normalize_run_script_step(step)
    pipeline_step: dict[str, Any] = {
        "id": f"step-{uuid4().hex[:8]}",
        "tool_name": normalized.get("tool_name", "unknown"),
        "parameters": normalized.get("parameters", {}),
        "description": normalized.get("description", f"Step {step_index + 1}"),
        "status": StepStatus.PENDING.value,
        "result": None,
        "error": None,
        "started_at": None,
        "completed_at": None,
    }

    generated_code = normalized.get("generated_code")
    if isinstance(generated_code, str) and generated_code.strip():
        pipeline_step["generated_code"] = generated_code.strip()

    return pipeline_step


def _generate_training_script(
    model_type: str,
    data_dir: str,
    parameters: dict[str, Any],
) -> str:
    """Generate a Python training script for the requested model type."""
    epochs = int(parameters.get("epochs", 50))
    batch_size = int(parameters.get("batch_size", 16))
    img_size = int(parameters.get("img_size", 640))

    # Normalize model identifier.
    model_id = model_type.lower().strip()
    if model_id in ("yolov8", "yolo", "ultralytics", "model"):
        model_id = "yolov8"

    classes = parameters.get("classes", [])
    if isinstance(classes, str):
        classes = [c.strip() for c in classes.split(",") if c.strip()]
    nc = len(classes) if classes else "len(class_names)"
    names_literal = repr(classes) if classes else "class_names"

    if model_id.startswith("yolov"):
        variant = model_id  # e.g. "yolov8", "yolov5"
        weight_file = f"{variant}n.pt"
        return (
            "#!/usr/bin/env python3\n"
            f'"""Auto-generated {variant.upper()} training script."""\n'
            "import os, yaml\n"
            "from pathlib import Path\n"
            "from ultralytics import YOLO\n"
            "\n"
            f'DATA_DIR = Path(r"{data_dir}")\n'
            "\n"
            "# --- Build data.yaml ------------------------------------------------\n"
            f"class_names = {names_literal}\n"
            "data_yaml = {\n"
            '    "path": str(DATA_DIR),\n'
            '    "train": "train/images",\n'
            '    "val": "val/images",\n'
            '    "test": "test/images",\n'
            f'    "nc": {nc},\n'
            '    "names": class_names,\n'
            "}\n"
            'yaml_path = DATA_DIR / "data.yaml"\n'
            'with open(yaml_path, "w") as f:\n'
            "    yaml.safe_dump(data_yaml, f)\n"
            'print(f"Wrote {yaml_path}")\n'
            "\n"
            "# --- Train -----------------------------------------------------------\n"
            f'model = YOLO("{weight_file}")\n'
            "results = model.train(\n"
            "    data=str(yaml_path),\n"
            f"    epochs={epochs},\n"
            f"    batch={batch_size},\n"
            f"    imgsz={img_size},\n"
            '    project=str(DATA_DIR / "runs"),\n'
            f'    name="{variant}-training",\n'
            ")\n"
            'print("Training complete. Best weights:", results.save_dir)\n'
        )

    # Fallback: generic PyTorch-style training script.
    return (
        "#!/usr/bin/env python3\n"
        f'"""Auto-generated training script for {model_type}."""\n'
        "# TODO: implement training loop for the requested model.\n"
        f'DATA_DIR = r"{data_dir}"\n'
        f"EPOCHS = {epochs}\n"
        f"BATCH_SIZE = {batch_size}\n"
        f"IMG_SIZE = {img_size}\n"
        f"print('Starting {model_type} training on', DATA_DIR)\n"
    )


def build_rule_based_plan(understanding: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Build a deterministic plan from parsed understanding.

    This fast-path avoids a second LLM call for common task-oriented workflows.
    Returns None if request is too ambiguous and should fall back to LLM planning.
    """
    input_path = understanding.get("input_path")
    operations = [
        str(op).strip().lower() for op in understanding.get("operations", []) if str(op).strip()
    ]
    parameters = dict(understanding.get("parameters", {}))
    output_path = understanding.get("output_path")

    if not input_path or not operations:
        return None

    # Keep scan-only flows in the LLM path for broader backwards compatibility.
    if len(operations) == 1 and operations[0] == "scan":
        return None

    # Ensure operation order is stable and valid.
    allowed = {
        "scan",
        "detect",
        "segment",
        "anonymize",
        "export",
        "convert_format",
        "label",
        "split",
        "find_duplicates",
        "label_qa",
        "script",
        "train",
        "review",
    }
    operations = [op for op in operations if op in allowed]
    if not operations:
        return None

    plan: list[dict[str, Any]] = [
        {
            "tool_name": "scan_directory",
            "parameters": {"path": input_path, "recursive": True},
            "description": "Scan input directory to verify contents and file formats",
        }
    ]

    # Labeling implies detect + export flow.
    if "label" in operations:
        if "detect" not in operations:
            operations.append("detect")
        if "export" not in operations:
            operations.append("export")

    working_path = input_path
    detection_annotations_path: str | None = None

    if "detect" in operations:
        classes = _normalize_classes(parameters.get("classes"))
        if not classes:
            classes = []
        confidence = float(parameters.get("confidence", 0.5))
        detection_annotations_path = f"{working_path}_detections_yolo"
        detect_parameters: dict[str, Any] = {
            "input_path": working_path,
            "confidence": confidence,
            "save_annotations": True,
            "output_path": detection_annotations_path,
        }
        if classes:
            detect_parameters["classes"] = classes
        plan.append(
            {
                "tool_name": "detect",
                "parameters": detect_parameters,
                "description": "Detect target objects in the input dataset",
            }
        )

    if "segment" in operations:
        prompt = str(parameters.get("prompt") or "objects of interest")
        plan.append(
            {
                "tool_name": "segment",
                "parameters": {
                    "input_path": working_path,
                    "prompt": prompt,
                },
                "description": "Segment requested objects in images",
            }
        )

    if "anonymize" in operations:
        anonymized_output = str(output_path or f"{input_path}_anonymized")
        target = str(parameters.get("target") or "all")
        plan.append(
            {
                "tool_name": "anonymize",
                "parameters": {
                    "input_path": working_path,
                    "output_path": anonymized_output,
                    "target": target,
                },
                "description": "Anonymize sensitive regions (faces and/or plates)",
            }
        )
        working_path = anonymized_output

    if "export" in operations:
        output_format = str(parameters.get("output_format") or parameters.get("format") or "yolo")
        export_output = str(output_path or f"{working_path}_{output_format}")
        export_source = detection_annotations_path or working_path
        export_parameters: dict[str, Any] = {
            "source_path": export_source,
            "output_path": export_output,
            "output_format": output_format,
        }
        if detection_annotations_path:
            export_parameters["source_format"] = "yolo"
        plan.append(
            {
                "tool_name": "export",
                "parameters": export_parameters,
                "description": f"Export annotations in {output_format.upper()} format",
            }
        )

    if "split" in operations:
        split_output = str(output_path or f"{working_path}_split")
        plan.append(
            {
                "tool_name": "split_dataset",
                "parameters": {
                    "path": working_path,
                    "output_path": split_output,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                },
                "description": "Split dataset into train/validation/test sets",
            }
        )
        working_path = split_output

    if "script" in operations or "train" in operations:
        script_params: dict[str, Any] = {
            "input_path": working_path,
        }
        generated_code: str | None = None
        if output_path:
            script_params["output_path"] = output_path

        # Generate actual script content based on what the user asked for.
        model_type = str(parameters.get("model_type", "yolov8"))
        is_training = parameters.get("training") or "train" in operations

        if is_training:
            # Build a training script for the detected model architecture.
            data_dir = working_path
            generated_code = _generate_training_script(model_type, data_dir, parameters)
            script_params["description"] = (
                f"Train {model_type} model on the prepared dataset"
            )
            description = f"Train {model_type} model on the prepared dataset"
        else:
            # Generic custom script — use the user's description or forward
            # any explicit script/command parameters.
            custom_desc = parameters.get("custom_step_description", "")
            if parameters.get("script"):
                script_value = str(parameters["script"])
                if _looks_like_inline_python_script(script_value):
                    generated_code = script_value.strip()
                else:
                    script_params["script"] = script_value
            if parameters.get("command"):
                script_params["command"] = parameters["command"]
            if custom_desc:
                description = custom_desc
                script_params["description"] = custom_desc
            else:
                description = "Execute custom processing script on the data"

        script_step: dict[str, Any] = {
            "tool_name": "run_script",
            "parameters": script_params,
            "description": description,
        }
        if generated_code:
            script_step["generated_code"] = generated_code
        plan.append(script_step)

    if "review" in operations:
        review_source = detection_annotations_path or working_path
        plan.append(
            {
                "tool_name": "review",
                "parameters": {
                    "source_path": review_source,
                    "image_dir": input_path,
                },
                "description": "Queue results for manual review in the Review Queue",
            }
        )

    if "convert_format" in operations:
        output_format = str(
            parameters.get("output_format") or parameters.get("format") or "yolo"
        )
        convert_output = str(output_path or f"{working_path}_{output_format}")
        plan.append(
            {
                "tool_name": "convert_format",
                "parameters": {
                    "source_path": working_path,
                    "output_path": convert_output,
                    "target_format": output_format,
                },
                "description": f"Convert annotations to {output_format.upper()} format",
            }
        )

    if "find_duplicates" in operations:
        plan.append(
            {
                "tool_name": "find_duplicates",
                "parameters": {
                    "path": working_path,
                    "method": str(parameters.get("method", "phash")),
                    "threshold": float(parameters.get("threshold", 0.9)),
                    "auto_remove": bool(parameters.get("auto_remove", False)),
                },
                "description": "Find and report duplicate or near-duplicate images",
            }
        )

    if "label_qa" in operations:
        qa_source = detection_annotations_path or working_path
        plan.append(
            {
                "tool_name": "label_qa",
                "parameters": {
                    "path": qa_source,
                    "generate_report": True,
                },
                "description": "Run quality-assurance checks on annotations",
            }
        )

    if len(plan) <= 1:
        return None

    return plan


def validate_plan(plan: list[dict[str, Any]]) -> str | None:
    """
    Validate a generated plan for correctness.

    Checks:
    - Plan is not empty
    - All tools are known
    - Each step has required parameters
    - Tool-specific parameter validation

    Args:
        plan: List of plan step dictionaries.

    Returns:
        Error message string if invalid, None if valid.
    """
    if not plan:
        return "Plan is empty"

    registered_tools = _get_registered_tools()
    valid_tools = set(VALID_TOOLS) | set(registered_tools)

    for i, step in enumerate(plan):
        step_num = i + 1

        # Check tool name exists and is valid
        tool_name = step.get("tool_name")
        if not tool_name:
            return f"Step {step_num} has no tool_name"

        if tool_name not in valid_tools:
            return f"Step {step_num} uses unknown tool: {tool_name}"

        # Check parameters shape
        parameters = step.get("parameters")
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            return f"Step {step_num} ({tool_name}) parameters must be an object"

        # Validate core planner expectations first for backwards compatibility.
        validation_error = _validate_core_step_parameters(step_num, tool_name, parameters)
        if validation_error:
            return validation_error

        # Keep compatibility for run_script/custom_script planning conventions.
        if tool_name in {"run_script", "custom_script"}:
            continue

        # Validate required parameters from registered tool metadata when available.
        tool = registered_tools.get(tool_name)
        if tool is not None:
            for param in getattr(tool, "parameters", []):
                if param.required and param.default is None and param.name not in parameters:
                    return (
                        f"Step {step_num} ({tool_name}) missing required "
                        f"'{param.name}' parameter"
                    )
            continue

        # Unknown runtime tool but known to planner constants: keep permissive.
        continue

    return None


def format_plan_for_display(plan: list[dict[str, Any]]) -> str:
    """
    Format a plan as a human-readable string for chat display.

    Args:
        plan: List of plan step dictionaries.

    Returns:
        Formatted string with numbered steps, tools, and parameters.
    """
    if not plan:
        return "No steps in plan."

    lines: list[str] = []

    for i, step in enumerate(plan, 1):
        status = step.get("status", StepStatus.PENDING.value)

        # Status icons
        status_icons = {
            StepStatus.PENDING.value: "[ ]",
            StepStatus.RUNNING.value: "[>]",
            StepStatus.COMPLETED.value: "[x]",
            StepStatus.FAILED.value: "[!]",
            StepStatus.SKIPPED.value: "[-]",
        }
        icon = status_icons.get(status, "[ ]")

        description = step.get("description", f"Step {i}")
        tool_name = step.get("tool_name", "unknown")
        parameters = step.get("parameters", {})

        lines.append(f"{icon} **Step {i}: {description}**")
        lines.append(f"    Tool: `{tool_name}`")

        if parameters:
            param_items = []
            for k, v in parameters.items():
                v_str = ", ".join(str(item) for item in v) if isinstance(v, list) else str(v)
                param_items.append(f"{k}={v_str}")
            lines.append(f"    Parameters: {', '.join(param_items)}")
        if step.get("generated_code"):
            lines.append("    Generated code: attached")

        lines.append("")  # Blank line between steps

    return "\n".join(lines)


async def generate_plan(state: PipelineState) -> dict[str, Any]:
    """
    Generate an execution plan from the understood request.

    This node:
    1. Gets the understanding from metadata
    2. Sends it to the LLM with the planning prompt
    3. Parses the JSON array response into pipeline steps
    4. Validates the plan
    5. Formats and displays the plan for user approval

    Args:
        state: Current pipeline state with understanding in metadata.

    Returns:
        State update dict with plan, or error message.
    """
    messages = state.get("messages", [])
    metadata = state.get("metadata", {})
    understanding = metadata.get("understanding", {})

    if not understanding:
        logger.warning("No understanding found in metadata")
        return {"last_error": "No understanding found, cannot plan"}

    logger.info(f"Generating plan for intent: {understanding.get('intent')}")

    # Fast-path deterministic planning for clear multi-operation workflows.
    rule_based_steps = build_rule_based_plan(understanding)
    if rule_based_steps is not None:
        plan = [_to_pipeline_step(step, i) for i, step in enumerate(rule_based_steps)]

        validation_error = validate_plan(plan)
        if validation_error is None:
            plan_display = format_plan_for_display(plan)
            plan_message: dict[str, Any] = {
                "role": MessageRole.ASSISTANT.value,
                "content": (
                    f"Here's my proposed plan:\n\n{plan_display}\n"
                    "Do you want to proceed, or would you like to make changes?"
                ),
                "timestamp": datetime.now().isoformat(),
                "tool_calls": [],
                "tool_call_id": None,
            }
            return {
                "messages": [*messages, plan_message],
                "plan": plan,
                "current_step": 0,
                "plan_approved": False,
            }

    # Load the planning prompt
    try:
        planning_prompt = load_prompt("planning")
    except FileNotFoundError:
        logger.error("Failed to load planning prompt")
        return {"last_error": "Failed to load planning prompt"}

    # Build the context for planning
    planning_context = f"""Create a plan for the following request:

Intent: {understanding.get("intent", "unknown")}
Input path: {understanding.get("input_path", "not specified")}
Input type: {understanding.get("input_type", "unknown")}
Operations: {understanding.get("operations", [])}
Parameters: {understanding.get("parameters", {})}
Output path: {understanding.get("output_path", "not specified")}

Generate a JSON array of steps to accomplish this."""

    llm_messages = [
        SystemMessage(content=planning_prompt),
        HumanMessage(content=planning_context),
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

            # Parse JSON array from response
            steps_raw = extract_json_array(str(response_content))

            if steps_raw is None:
                last_error = f"Failed to parse JSON array from response: {response_content[:200]}"
                logger.warning(last_error)
                continue

            # Convert to PipelineStep format
            plan = [_to_pipeline_step(step, i) for i, step in enumerate(steps_raw)]

            # Validate the plan
            validation_error = validate_plan(plan)
            if validation_error:
                logger.warning(f"Plan validation failed: {validation_error}")

                # Try to fix common issues on retry
                last_error = validation_error
                if attempt < MAX_RETRIES - 1:
                    # Add validation error to context for next attempt
                    llm_messages.append(
                        HumanMessage(
                            content=f"The plan had an error: {validation_error}. "
                            "Please fix and try again."
                        )
                    )
                    continue
                else:
                    # Final attempt failed, report error
                    error_message: dict[str, Any] = {
                        "role": MessageRole.ASSISTANT.value,
                        "content": f"I had trouble creating a valid plan: {validation_error}. Could you rephrase your request?",
                        "timestamp": datetime.now().isoformat(),
                        "tool_calls": [],
                        "tool_call_id": None,
                    }
                    return {
                        "messages": [*messages, error_message],
                        "last_error": validation_error,
                        "awaiting_user": True,
                    }

            # Plan is valid, format for display
            plan_display = format_plan_for_display(plan)
            logger.info(f"Generated plan with {len(plan)} steps")

            plan_message: dict[str, Any] = {
                "role": MessageRole.ASSISTANT.value,
                "content": (
                    f"Here's my proposed plan:\n\n{plan_display}\n"
                    "Do you want to proceed, or would you like to make changes?"
                ),
                "timestamp": datetime.now().isoformat(),
                "tool_calls": [],
                "tool_call_id": None,
            }

            return {
                "messages": [*messages, plan_message],
                "plan": plan,
                "current_step": 0,
                "plan_approved": False,
            }

        except Exception as e:
            last_error = f"LLM call failed: {e}"
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                break

    # All retries failed
    logger.error(f"All {MAX_RETRIES} attempts failed: {last_error}")
    error_message = {
        "role": MessageRole.ASSISTANT.value,
        "content": "I had trouble creating a structured plan. Could you rephrase your request?",
        "timestamp": datetime.now().isoformat(),
        "tool_calls": [],
        "tool_call_id": None,
    }

    return {
        "messages": [*messages, error_message],
        "last_error": last_error,
        "awaiting_user": True,
    }


__all__ = [
    "generate_plan",
    "build_rule_based_plan",
    "validate_plan",
    "format_plan_for_display",
    "VALID_TOOLS",
]
