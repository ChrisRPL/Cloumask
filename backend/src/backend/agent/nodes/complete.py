"""
Complete node for the Cloumask agent.

This module implements the complete_node that finalizes pipeline execution,
generates summaries, and aggregates statistics.

Implements spec: 04-agent-nodes-execution
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from backend.agent.state import MessageRole, PipelineState

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Statistics Calculation
# -----------------------------------------------------------------------------


def calculate_final_stats(
    plan: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate aggregate statistics for the pipeline.

    Args:
        plan: List of pipeline steps with status information.
        results: Dictionary of step_id -> result mappings.

    Returns:
        Dictionary of final statistics.
    """
    # Count step statuses
    completed = sum(1 for s in plan if s.get("status") == "completed")
    failed = sum(1 for s in plan if s.get("status") == "failed")
    skipped = sum(1 for s in plan if s.get("status") == "skipped")

    # Calculate total duration
    total_duration = 0.0
    for step in plan:
        started_at = step.get("started_at")
        completed_at = step.get("completed_at")
        if started_at and completed_at:
            try:
                start = datetime.fromisoformat(started_at)
                end = datetime.fromisoformat(completed_at)
                total_duration += (end - start).total_seconds()
            except (ValueError, TypeError):
                pass

    # Aggregate counts from results
    files_processed = 0
    items_detected = 0
    items_anonymized = 0
    masks_generated = 0

    for result in results.values():
        if isinstance(result, dict) and "error" not in result:
            files_processed += result.get("files_processed", 0)
            items_detected += result.get("count", 0)
            items_anonymized += (
                result.get("faces_blurred", 0) + result.get("plates_blurred", 0)
            )
            masks_generated += result.get("masks_generated", 0)

    # Calculate average confidence if available
    confidences: list[float] = []
    for result in results.values():
        if isinstance(result, dict) and "confidence" in result:
            conf = result["confidence"]
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))

    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    total_steps = len(plan)
    return {
        "total_steps": total_steps,
        "completed_steps": completed,
        "failed_steps": failed,
        "skipped_steps": skipped,
        "success_rate": completed / total_steps if total_steps > 0 else 0,
        "total_duration_seconds": total_duration,
        "files_processed": files_processed,
        "items_detected": items_detected,
        "items_anonymized": items_anonymized,
        "masks_generated": masks_generated,
        "average_confidence": avg_confidence,
    }


# -----------------------------------------------------------------------------
# Summary Generation
# -----------------------------------------------------------------------------


def generate_summary(
    plan: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    """
    Generate human-readable completion summary.

    Args:
        plan: List of pipeline steps.
        results: Dictionary of step results.
        stats: Calculated statistics.

    Returns:
        Markdown-formatted summary string.
    """
    lines = ["## Pipeline Complete"]
    lines.append("")

    # Overall status
    failed_steps = stats.get("failed_steps", 0)
    if failed_steps == 0:
        lines.append("All steps completed successfully.")
    else:
        lines.append(f"Completed with {failed_steps} failed step(s).")

    lines.append("")
    lines.append("### Summary")
    lines.append(
        f"- Steps: {stats['completed_steps']}/{stats['total_steps']} completed"
    )

    duration = stats.get("total_duration_seconds", 0)
    if duration > 0:
        lines.append(f"- Duration: {duration:.1f}s")

    if stats.get("files_processed", 0) > 0:
        lines.append(f"- Files processed: {stats['files_processed']}")

    if stats.get("items_detected", 0) > 0:
        lines.append(f"- Objects detected: {stats['items_detected']}")

    if stats.get("items_anonymized", 0) > 0:
        lines.append(f"- Items anonymized: {stats['items_anonymized']}")

    if stats.get("masks_generated", 0) > 0:
        lines.append(f"- Masks generated: {stats['masks_generated']}")

    if stats.get("average_confidence"):
        lines.append(f"- Avg confidence: {stats['average_confidence']:.1%}")

    # List any failures
    failed_step_list = [s for s in plan if s.get("status") == "failed"]
    if failed_step_list:
        lines.append("")
        lines.append("### Failed Steps")
        for step in failed_step_list:
            description = step.get("description", step.get("tool_name", "Unknown"))
            error = step.get("error", "Unknown error")
            lines.append(f"- {description}: {error}")

    # Output location if available
    export_results = [
        r for r in results.values()
        if isinstance(r, dict) and "output_path" in r
    ]
    if export_results:
        lines.append("")
        lines.append("### Output")
        for r in export_results:
            lines.append(f"- {r['output_path']}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Complete Node
# -----------------------------------------------------------------------------


async def complete_node(state: PipelineState) -> dict[str, Any]:
    """
    Finalize pipeline execution.

    Generates a completion summary, aggregates statistics, and
    adds the final message to the conversation.

    Updates state with:
    - Final summary message
    - Aggregated statistics in metadata
    - Completion timestamp

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with summary and statistics.
    """
    plan = state.get("plan", [])
    results = state.get("execution_results", {})
    metadata = dict(state.get("metadata", {}))
    messages = list(state.get("messages", []))

    logger.info("Completing pipeline execution")

    # Calculate final statistics
    stats = calculate_final_stats(plan, results)
    metadata["final_stats"] = stats
    metadata["completed_at"] = datetime.now().isoformat()

    logger.info(
        "Pipeline stats: %d/%d completed, %d failed, %.1fs total",
        stats["completed_steps"],
        stats["total_steps"],
        stats["failed_steps"],
        stats["total_duration_seconds"],
    )

    # Generate summary message
    summary = generate_summary(plan, results, stats)

    messages.append({
        "role": MessageRole.ASSISTANT.value,
        "content": summary,
        "timestamp": datetime.now().isoformat(),
        "tool_calls": [],
        "tool_call_id": None,
    })

    return {
        "metadata": metadata,
        "messages": messages,
        "awaiting_user": False,
    }


# Alias for backward compatibility
complete = complete_node
