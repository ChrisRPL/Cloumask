"""Compatibility tests for custom script tool aliases."""

from backend.agent.tools import get_tool_registry, initialize_tools


def test_run_script_alias_is_registered() -> None:
    """Planner legacy step name should resolve to the custom script tool."""
    initialize_tools()
    registry = get_tool_registry()

    assert registry.has("custom_script")
    assert registry.has("run_script")
