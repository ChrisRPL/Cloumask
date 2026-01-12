"""
Human-in-the-Loop approval node for the Cloumask agent.

This module implements the await_approval node that pauses execution for human
review, along with helper functions for processing user responses and applying
plan edits.

Key components:
- await_approval_node: Pauses execution and marks state as awaiting user input
- handle_user_response: Processes user decisions (approve, edit, cancel, retry)
- apply_plan_edits: Applies user modifications to the execution plan
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from backend.agent.state import PipelineState, StepStatus, UserDecision

logger = logging.getLogger(__name__)


def await_approval_node(state: PipelineState) -> dict[str, Any]:
    """
    Pause execution and wait for user decision.

    This node marks the state as awaiting user input. The actual user response
    is handled by the API layer, which calls handle_user_response to update
    the state and resume the graph.

    Determines what we're waiting for based on state:
    - If plan not approved: waiting for initial plan approval
    - If checkpoints exist with unresolved checkpoint: waiting for checkpoint approval

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with:
        - awaiting_user: True
        - messages: Updated with appropriate system message
    """
    plan_approved = state.get("plan_approved", False)
    checkpoints = state.get("checkpoints", [])
    existing_messages = list(state.get("messages", []))

    if not plan_approved:
        # Waiting for initial plan approval
        logger.info("Awaiting plan approval from user")
        existing_messages.append({
            "role": "system",
            "content": "AWAIT_PLAN_APPROVAL",
            "timestamp": datetime.now().isoformat(),
        })
        return {
            "awaiting_user": True,
            "messages": existing_messages,
        }

    # Waiting for checkpoint approval
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        checkpoint_id = latest_checkpoint.get("id", "unknown")

        # Only wait if checkpoint is unresolved
        if not latest_checkpoint.get("resolved_at"):
            logger.info(f"Awaiting checkpoint approval: {checkpoint_id}")
            existing_messages.append({
                "role": "system",
                "content": f"AWAIT_CHECKPOINT:{checkpoint_id}",
                "timestamp": datetime.now().isoformat(),
            })
            return {
                "awaiting_user": True,
                "messages": existing_messages,
            }

    # No specific approval needed, just set awaiting flag
    logger.info("Awaiting user input")
    return {"awaiting_user": True}


def handle_user_response(
    state: PipelineState,
    response: dict[str, Any],
) -> PipelineState:
    """
    Process user's decision and update state accordingly.

    Called by API layer when user responds to an approval request.
    Handles four decision types:
    - approve: Continue execution
    - edit: Apply plan modifications
    - cancel: Skip remaining steps and complete
    - retry: Retry the current step

    Args:
        state: Current pipeline state.
        response: User response dict with:
            - decision: "approve" | "edit" | "cancel" | "retry"
            - message: Optional feedback message
            - plan_edits: Optional list of plan modifications

    Returns:
        Updated PipelineState with user decision applied.
    """
    # Create a mutable copy of the state
    updated_state: PipelineState = dict(state)  # type: ignore[assignment]

    decision_str = response.get("decision", "approve")
    try:
        decision = UserDecision(decision_str)
    except ValueError:
        logger.warning(f"Invalid decision '{decision_str}', defaulting to approve")
        decision = UserDecision.APPROVE

    message = response.get("message")
    plan_edits = response.get("plan_edits")

    # Get mutable copies of lists
    messages = list(updated_state.get("messages", []))
    checkpoints = list(updated_state.get("checkpoints", []))
    plan = list(updated_state.get("plan", []))

    logger.info(f"Processing user decision: {decision.value}")

    # Add user message to conversation if provided
    if message:
        messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

    # Handle based on decision type
    if decision == UserDecision.APPROVE:
        updated_state["awaiting_user"] = False
        updated_state["plan_approved"] = True

        # Resolve any pending checkpoint
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            if not last_checkpoint.get("resolved_at"):
                last_checkpoint["user_decision"] = decision.value
                last_checkpoint["resolved_at"] = datetime.now().isoformat()
                last_checkpoint["user_feedback"] = message

        messages.append({
            "role": "assistant",
            "content": "Continuing execution...",
            "timestamp": datetime.now().isoformat(),
        })
        logger.info("User approved - continuing execution")

    elif decision == UserDecision.EDIT:
        updated_state["awaiting_user"] = False
        # Edited plans require re-approval
        updated_state["plan_approved"] = False

        if plan_edits:
            # Apply the edits to the plan
            plan = apply_plan_edits(plan, plan_edits)
            updated_state["plan"] = plan

            messages.append({
                "role": "assistant",
                "content": "Plan updated. Here's the revised plan:",
                "timestamp": datetime.now().isoformat(),
            })
            logger.info(f"Applied {len(plan_edits)} plan edits")
        else:
            # No edits provided, need to wait for them
            messages.append({
                "role": "assistant",
                "content": "What changes would you like to make to the plan?",
                "timestamp": datetime.now().isoformat(),
            })
            updated_state["awaiting_user"] = True
            logger.info("Awaiting plan edit details from user")

    elif decision == UserDecision.CANCEL:
        updated_state["awaiting_user"] = False

        messages.append({
            "role": "assistant",
            "content": "Pipeline cancelled.",
            "timestamp": datetime.now().isoformat(),
        })

        # Mark remaining steps as skipped
        current_step = updated_state.get("current_step", 0)
        for i, step in enumerate(plan):
            if i >= current_step:
                step["status"] = StepStatus.SKIPPED.value

        updated_state["plan"] = plan
        logger.info(f"Pipeline cancelled - skipped {len(plan) - current_step} steps")

    elif decision == UserDecision.RETRY:
        updated_state["awaiting_user"] = False
        updated_state["retry_count"] = 0

        # Resolve any pending checkpoint
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            if not last_checkpoint.get("resolved_at"):
                last_checkpoint["user_decision"] = decision.value
                last_checkpoint["resolved_at"] = datetime.now().isoformat()
                last_checkpoint["user_feedback"] = message

        messages.append({
            "role": "assistant",
            "content": "Retrying the current step...",
            "timestamp": datetime.now().isoformat(),
        })
        logger.info("User requested retry - resetting retry count")

    updated_state["messages"] = messages
    updated_state["checkpoints"] = checkpoints

    return updated_state


def apply_plan_edits(
    plan: list[dict[str, Any]],
    edits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply user's edits to the execution plan.

    Supported edit actions:
    - modify: Update step parameters or properties
    - remove: Delete a step from the plan
    - add: Insert a new step after a specified index
    - reorder: Move a step to a different position

    Args:
        plan: Current execution plan as list of step dicts.
        edits: List of edit operations to apply:
            - {"action": "modify", "step_index": 0, "changes": {"parameters": {...}}}
            - {"action": "remove", "step_index": 2}
            - {"action": "add", "after_index": 1, "step": {...}}
            - {"action": "reorder", "from_index": 3, "to_index": 1}

    Returns:
        Modified plan with edits applied.

    Note:
        Edits are applied in order, so indices may shift after remove/add operations.
        Invalid indices are silently ignored to prevent crashes.
    """
    # Work with a copy to avoid mutation
    modified_plan = [dict(step) for step in plan]

    for edit in edits:
        action = edit.get("action")

        if action == "modify":
            idx = edit.get("step_index")
            if idx is not None and 0 <= idx < len(modified_plan):
                changes = edit.get("changes", {})
                # Deep update the step
                for key, value in changes.items():
                    if key == "parameters" and isinstance(value, dict):
                        # Merge parameters rather than replace
                        existing_params = modified_plan[idx].get("parameters", {})
                        existing_params.update(value)
                        modified_plan[idx]["parameters"] = existing_params
                    else:
                        modified_plan[idx][key] = value
                logger.debug(f"Modified step {idx}: {list(changes.keys())}")

        elif action == "remove":
            idx = edit.get("step_index")
            if idx is not None and 0 <= idx < len(modified_plan):
                removed = modified_plan.pop(idx)
                logger.debug(f"Removed step {idx}: {removed.get('tool_name')}")

        elif action == "add":
            after_idx = edit.get("after_index", len(modified_plan) - 1)
            new_step = edit.get("step")
            # Validate new step exists and has required tool_name field
            if new_step and "tool_name" in new_step:
                # Copy to avoid mutating the input
                new_step = dict(new_step)
                # Ensure required fields exist with unique IDs
                if "id" not in new_step:
                    new_step["id"] = f"step-{uuid4().hex[:8]}"
                if "status" not in new_step:
                    new_step["status"] = StepStatus.PENDING.value
                if "description" not in new_step:
                    new_step["description"] = f"Execute {new_step['tool_name']}"

                insert_idx = min(after_idx + 1, len(modified_plan))
                modified_plan.insert(insert_idx, new_step)
                logger.debug(
                    f"Added step at {insert_idx}: {new_step.get('tool_name')}"
                )

        elif action == "reorder":
            from_idx = edit.get("from_index")
            to_idx = edit.get("to_index")
            if (
                from_idx is not None
                and to_idx is not None
                and 0 <= from_idx < len(modified_plan)
                and 0 <= to_idx < len(modified_plan)
            ):
                step = modified_plan.pop(from_idx)
                modified_plan.insert(to_idx, step)
                logger.debug(
                    f"Reordered step from {from_idx} to {to_idx}: "
                    f"{step.get('tool_name')}"
                )

    return modified_plan


__all__ = [
    "await_approval_node",
    "handle_user_response",
    "apply_plan_edits",
]
