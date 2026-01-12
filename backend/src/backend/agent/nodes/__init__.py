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

- spec 05-human-in-the-loop:
  - await_approval: Pause execution for user review
  - create_checkpoint: Evaluate quality and create checkpoint if needed
  - handle_user_response: Process user decisions
  - apply_plan_edits: Apply user modifications to the plan
"""

from __future__ import annotations

from typing import Any

# Spec 05 - Human-in-the-Loop nodes
from backend.agent.nodes.approval import (
    apply_plan_edits,
    await_approval_node,
    handle_user_response,
)
from backend.agent.nodes.checkpoint import (
    CONFIDENCE_DROP_THRESHOLD,
    CRITICAL_TOOLS,
    ERROR_RATE_THRESHOLD,
    PERCENTAGE_THRESHOLDS,
    calculate_quality_metrics,
    checkpoint_node,
    determine_trigger,
    format_checkpoint_message,
)

# Spec 04 - Execution nodes
from backend.agent.nodes.complete import (
    calculate_final_stats,
    complete,
    complete_node,
    generate_summary,
)
from backend.agent.nodes.execute import (
    StubTool,
    execute_step,
    execute_step_node,
    format_step_result,
    is_retryable,
    register_stub_tools,
    update_progress,
)

# Import real implementations from submodules
# Spec 03 - Planning nodes
from backend.agent.nodes.plan import (
    VALID_TOOLS,
    format_plan_for_display,
    generate_plan,
    validate_plan,
)
from backend.agent.nodes.understand import understand
from backend.agent.state import PipelineState

# Re-export from tools package (spec 06)
from backend.agent.tools import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
)


# Alias functions for graph compatibility
# The graph uses 'await_approval' and 'create_checkpoint' as node names
def await_approval(state: PipelineState) -> dict[str, Any]:
    """Alias for await_approval_node for graph compatibility."""
    return await_approval_node(state)


def create_checkpoint(state: PipelineState) -> dict[str, Any]:
    """Alias for checkpoint_node for graph compatibility."""
    return checkpoint_node(state)


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
    "StubTool",
    "register_stub_tools",
    # Tool system (spec 06)
    "BaseTool",
    "ToolCategory",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "get_tool_registry",
    # Human-in-the-Loop nodes (spec 05)
    "await_approval",
    "await_approval_node",
    "create_checkpoint",
    "checkpoint_node",
    "handle_user_response",
    "apply_plan_edits",
    "determine_trigger",
    "calculate_quality_metrics",
    "format_checkpoint_message",
    "CRITICAL_TOOLS",
    "PERCENTAGE_THRESHOLDS",
    "ERROR_RATE_THRESHOLD",
    "CONFIDENCE_DROP_THRESHOLD",
]
