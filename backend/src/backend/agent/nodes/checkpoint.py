"""
Checkpoint node for the Cloumask agent.

This module implements the checkpoint node that evaluates execution quality
and creates checkpoints when certain conditions are met, enabling human-in-the-loop
review at critical points in the pipeline.

Key components:
- checkpoint_node: Evaluates quality and creates checkpoint if needed
- determine_trigger: Determines why a checkpoint should be triggered
- calculate_quality_metrics: Calculates quality metrics from execution results
- format_checkpoint_message: Formats checkpoint notification for display
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from backend.agent.state import (
    CheckpointTrigger,
    PipelineState,
    QualityMetrics,
)

logger = logging.getLogger(__name__)

# Critical tools that always trigger a checkpoint after execution
CRITICAL_TOOLS = ["anonymize", "segment", "detect_3d"]

# Percentage thresholds for progress-based checkpoints
PERCENTAGE_THRESHOLDS = [0.10, 0.25, 0.50]

# Quality thresholds
ERROR_RATE_THRESHOLD = 0.05  # 5% error rate
CONFIDENCE_DROP_THRESHOLD = 0.15  # 15% confidence drop


def checkpoint_node(state: PipelineState) -> dict[str, Any]:
    """
    Evaluate execution quality and create checkpoint if needed.

    This node analyzes the current execution state and creates a checkpoint
    record when trigger conditions are met. The checkpoint includes quality
    metrics and a notification message for the user.

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with:
        - checkpoints: Updated with new checkpoint if triggered
        - messages: Updated with checkpoint notification
        - awaiting_user: True if checkpoint requires user review
    """
    # Determine if and why a checkpoint should be created
    trigger = determine_trigger(state)

    if not trigger:
        # No checkpoint needed, continue execution
        logger.debug("No checkpoint trigger condition met")
        return {}

    # Calculate quality metrics
    metrics = calculate_quality_metrics(state)

    current_step = state.get("current_step", 0)
    existing_checkpoints = list(state.get("checkpoints", []))
    existing_messages = list(state.get("messages", []))

    # Create checkpoint record
    checkpoint = {
        "id": f"ckpt-{uuid4().hex[:8]}",
        "step_index": current_step,
        "trigger_reason": trigger.value,
        "quality_metrics": {
            "average_confidence": metrics.average_confidence,
            "error_count": metrics.error_count,
            "total_processed": metrics.total_processed,
            "processing_speed": metrics.processing_speed,
        },
        "created_at": datetime.now().isoformat(),
        "user_decision": None,
        "user_feedback": None,
        "resolved_at": None,
    }

    existing_checkpoints.append(checkpoint)
    logger.info(
        f"Created checkpoint {checkpoint['id']} at step {current_step} "
        f"(trigger: {trigger.value})"
    )

    # Generate checkpoint message for display
    message_content = format_checkpoint_message(trigger, metrics, state)
    existing_messages.append({
        "role": "assistant",
        "content": message_content,
        "timestamp": datetime.now().isoformat(),
    })

    # Determine if this checkpoint requires user review
    # Quality issues and critical steps always require review
    requires_review = trigger in (
        CheckpointTrigger.QUALITY_DROP,
        CheckpointTrigger.ERROR_RATE,
        CheckpointTrigger.CRITICAL_STEP,
    )

    return {
        "checkpoints": existing_checkpoints,
        "messages": existing_messages,
        "awaiting_user": requires_review,
    }


def determine_trigger(state: PipelineState) -> CheckpointTrigger | None:
    """
    Determine if and why a checkpoint should be created.

    Evaluates multiple trigger conditions in priority order:
    1. Critical step completion (always triggers)
    2. Quality drop (confidence decreased significantly)
    3. Error rate exceeded threshold
    4. Percentage milestone reached

    Args:
        state: Current pipeline state.

    Returns:
        CheckpointTrigger if checkpoint needed, None otherwise.
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    results = state.get("execution_results", {})

    if not plan or current_step == 0:
        return None

    total_steps = len(plan)
    progress = current_step / total_steps

    # Get existing checkpoint step indices to avoid duplicates
    existing_checkpoints = state.get("checkpoints", [])
    checkpoint_steps = {cp.get("step_index", -1) for cp in existing_checkpoints}

    # Already checkpointed at this step?
    if current_step in checkpoint_steps:
        return None

    # 1. Check for critical step completion (highest priority)
    last_step = plan[current_step - 1] if current_step > 0 else None
    if last_step:
        tool_name = last_step.get("tool_name", "")
        if tool_name in CRITICAL_TOOLS:
            logger.debug(f"Critical tool trigger: {tool_name}")
            return CheckpointTrigger.CRITICAL_STEP

        # Also check for explicitly marked critical steps
        if last_step.get("critical", False):
            logger.debug("Explicitly marked critical step")
            return CheckpointTrigger.CRITICAL_STEP

    # 2. Check for quality issues in execution results
    if results:
        result_values = list(results.values())
        recent_results = result_values[-5:] if len(result_values) >= 5 else result_values

        # Error rate check
        error_count = sum(
            1
            for r in recent_results
            if isinstance(r, dict) and r.get("error")
        )
        if len(recent_results) > 0:
            recent_error_rate = error_count / len(recent_results)
            if recent_error_rate > ERROR_RATE_THRESHOLD:
                logger.debug(f"Error rate trigger: {recent_error_rate:.1%}")
                return CheckpointTrigger.ERROR_RATE

        # Confidence drop check
        all_confidences: list[float] = [
            float(r["confidence"])
            for r in result_values
            if isinstance(r, dict) and "confidence" in r and r["confidence"] is not None
        ]
        recent_confidences: list[float] = [
            float(r["confidence"])
            for r in recent_results
            if isinstance(r, dict) and "confidence" in r and r["confidence"] is not None
        ]

        if all_confidences and recent_confidences:
            overall_avg = sum(all_confidences) / len(all_confidences)
            recent_avg = sum(recent_confidences) / len(recent_confidences)

            if overall_avg - recent_avg > CONFIDENCE_DROP_THRESHOLD:
                logger.debug(
                    f"Confidence drop trigger: overall={overall_avg:.2f}, "
                    f"recent={recent_avg:.2f}"
                )
                return CheckpointTrigger.QUALITY_DROP

    # 3. Check for percentage-based triggers
    prev_progress = (current_step - 1) / total_steps if current_step > 0 else 0

    for threshold in PERCENTAGE_THRESHOLDS:
        # Check if we just crossed this threshold
        if prev_progress < threshold <= progress:
            # Verify we haven't already checkpointed near this threshold
            threshold_step = int(threshold * total_steps)
            if threshold_step not in checkpoint_steps:
                logger.debug(f"Percentage trigger: {threshold:.0%}")
                return CheckpointTrigger.PERCENTAGE

    return None


def calculate_quality_metrics(state: PipelineState) -> QualityMetrics:
    """
    Calculate current quality metrics from execution results.

    Aggregates metrics from all executed steps including:
    - Average confidence scores across all results
    - Total error count
    - Total items processed
    - Processing speed (items per second)

    Args:
        state: Current pipeline state.

    Returns:
        QualityMetrics instance with calculated values.
    """
    results = state.get("execution_results", {})
    plan = state.get("plan", [])

    confidences: list[float] = []
    error_count = 0
    total_processed = 0

    for result in results.values():
        if not isinstance(result, dict):
            continue

        # Collect confidence scores
        if "confidence" in result:
            conf = result["confidence"]
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))

        # Count errors
        if result.get("error"):
            error_count += 1

        # Count processed items
        total_processed += result.get("files_processed", 0)
        total_processed += result.get("items_processed", 0)

        # If no specific count, assume 1 item per step
        if (
            "files_processed" not in result
            and "items_processed" not in result
            and not result.get("error")
        ):
            total_processed += 1

    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Calculate processing speed from step durations
    total_duration_seconds = 0.0
    for step in plan:
        started_at = step.get("started_at")
        completed_at = step.get("completed_at")

        if started_at and completed_at:
            try:
                start = datetime.fromisoformat(started_at)
                end = datetime.fromisoformat(completed_at)
                total_duration_seconds += (end - start).total_seconds()
            except (ValueError, TypeError):
                pass

    processing_speed = (
        total_processed / total_duration_seconds
        if total_duration_seconds > 0
        else 0.0
    )

    return QualityMetrics(
        average_confidence=avg_confidence,
        error_count=error_count,
        total_processed=total_processed,
        processing_speed=processing_speed,
    )


def format_checkpoint_message(
    trigger: CheckpointTrigger,
    metrics: QualityMetrics,
    state: PipelineState,
) -> str:
    """
    Format checkpoint notification for display to the user.

    Creates a markdown-formatted message with:
    - Checkpoint header with trigger reason
    - Progress information
    - Quality metrics summary
    - Trigger-specific guidance
    - Available user options

    Args:
        trigger: The reason the checkpoint was triggered.
        metrics: Current quality metrics.
        state: Current pipeline state for context.

    Returns:
        Formatted checkpoint message string.
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    total_steps = len(plan) if plan else 1
    progress = (current_step / total_steps) * 100 if total_steps > 0 else 0

    lines = [f"## Checkpoint ({trigger.value})"]
    lines.append("")
    lines.append(f"**Progress:** {progress:.0f}% ({current_step}/{total_steps} steps)")
    lines.append("")

    # Quality metrics section
    lines.append("### Quality Metrics")
    lines.append(f"- Avg Confidence: {metrics.average_confidence:.1%}")
    lines.append(f"- Error Rate: {metrics.error_rate:.1%}")
    lines.append(f"- Processing Speed: {metrics.processing_speed:.1f} items/sec")
    lines.append(f"- Total Processed: {metrics.total_processed}")

    # Trigger-specific message
    lines.append("")

    if trigger == CheckpointTrigger.PERCENTAGE:
        lines.append(f"This is a progress checkpoint at {progress:.0f}%.")
        lines.append("Review the results so far and decide if you want to continue.")

    elif trigger == CheckpointTrigger.CRITICAL_STEP:
        # Get the last completed tool
        last_tool = "unknown"
        if current_step > 0 and plan:
            last_step = plan[current_step - 1]
            last_tool = last_step.get("tool_name", "unknown")

        lines.append(f"Just completed critical operation: **{last_tool}**")
        lines.append("Please verify the results before continuing.")

    elif trigger == CheckpointTrigger.QUALITY_DROP:
        lines.append("**Warning:** Significant drop in confidence scores detected.")
        lines.append("Consider reviewing recent results or adjusting parameters.")

    elif trigger == CheckpointTrigger.ERROR_RATE:
        lines.append("**Warning:** Error rate is above 5%.")
        lines.append("Some items may not be processing correctly.")

    lines.append("")
    lines.append("**Options:** Continue | Edit Plan | Cancel")

    return "\n".join(lines)


__all__ = [
    "checkpoint_node",
    "determine_trigger",
    "calculate_quality_metrics",
    "format_checkpoint_message",
    "CRITICAL_TOOLS",
    "PERCENTAGE_THRESHOLDS",
    "ERROR_RATE_THRESHOLD",
    "CONFIDENCE_DROP_THRESHOLD",
]
