"""
LangGraph node implementations for the Cloumask agent.

This module provides node implementations for the agent state machine.

Implemented nodes:
- spec 03-agent-nodes-planning:
  - understand: Parse natural language request and extract intent
  - generate_plan: Generate execution plan using LLM

- spec 04-agent-nodes-execution:
  - execute_step: Execute current pipeline step
  - complete: Finalize pipeline and generate summary

Stub implementations for future specs:
- 05-human-in-the-loop: await_approval, checkpoint nodes
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

# Import real implementations from submodules
# Spec 03 - Planning nodes
from backend.agent.nodes.plan import (
    VALID_TOOLS,
    format_plan_for_display,
    generate_plan,
    validate_plan,
)
from backend.agent.nodes.understand import understand

# Spec 04 - Execution nodes
from backend.agent.nodes.complete import (
    calculate_final_stats,
    complete,
    complete_node,
    generate_summary,
)
from backend.agent.nodes.execute import (
    Tool,
    ToolRegistry,
    execute_step,
    execute_step_node,
    format_step_result,
    get_tool_registry,
    is_retryable,
    register_stub_tools,
    update_progress,
)
from backend.agent.state import CheckpointTrigger, PipelineState

# -----------------------------------------------------------------------------
# Stub implementations for future specs
# -----------------------------------------------------------------------------


def await_approval(state: PipelineState) -> dict[str, Any]:
    """
    Wait for human approval of plan or checkpoint.

    Sets awaiting_user flag to pause graph execution.

    Future implementation (spec 05):
    - Present plan/checkpoint to user via SSE
    - Wait for user feedback
    - Process approval/edit/cancel decision

    Args:
        state: Current pipeline state.

    Returns:
        State update with awaiting_user=True.
    """
    return {"awaiting_user": True}


def create_checkpoint(state: PipelineState) -> dict[str, Any]:
    """
    Create a human-in-the-loop checkpoint.

    Stub: Creates minimal checkpoint record.

    Future implementation (spec 05):
    - Calculate quality metrics
    - Determine checkpoint trigger reason
    - Create detailed checkpoint record
    - Set awaiting_user for critical checkpoints

    Args:
        state: Current pipeline state.

    Returns:
        State update with new checkpoint.
    """
    current_step = state.get("current_step", 0)
    existing_checkpoints = state.get("checkpoints", [])

    new_checkpoint = {
        "id": str(uuid4()),
        "step_index": current_step,
        "trigger_reason": CheckpointTrigger.PERCENTAGE.value,
        "quality_metrics": {
            "average_confidence": 1.0,
            "error_count": 0,
            "total_processed": current_step,
            "processing_speed": 0.0,
        },
        "created_at": datetime.now().isoformat(),
        "user_decision": None,
        "user_feedback": None,
        "resolved_at": None,
    }

    return {
        "checkpoints": [*existing_checkpoints, new_checkpoint],
        "awaiting_user": True,
    }


__all__ = [
    # Planning nodes (spec 03)
    "understand",
    "generate_plan",
    "validate_plan",
    "format_plan_for_display",
    "VALID_TOOLS",
    # Execution nodes (spec 04)
    "execute_step",
    "execute_step_node",
    "complete",
    "complete_node",
    "format_step_result",
    "update_progress",
    "is_retryable",
    "calculate_final_stats",
    "generate_summary",
    "Tool",
    "ToolRegistry",
    "get_tool_registry",
    "register_stub_tools",
    # Stubs (spec 05)
    "await_approval",
    "create_checkpoint",
]
