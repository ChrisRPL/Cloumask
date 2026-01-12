"""
Tests for the tool registry and discovery system.

Tests ToolRegistry singleton, registration, lookup, and decorators.

Implements spec: 06-tool-system (testing section)
"""

from __future__ import annotations

import pytest

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolResult,
    success_result,
)
from backend.agent.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    register_tool,
)

# -----------------------------------------------------------------------------
# Sample Tools (not prefixed with Test to avoid pytest collection)
# -----------------------------------------------------------------------------


class SampleToolA(BaseTool):
    """Sample tool A for registry tests."""

    name = "sample_tool_a"
    description = "Sample tool A"
    category = ToolCategory.SCAN
    parameters = []

    async def execute(self, **kwargs: object) -> ToolResult:
        return success_result({"name": "a"})


class SampleToolB(BaseTool):
    """Sample tool B for registry tests."""

    name = "sample_tool_b"
    description = "Sample tool B"
    category = ToolCategory.DETECTION
    parameters = []

    async def execute(self, **kwargs: object) -> ToolResult:
        return success_result({"name": "b"})


class SampleToolScan(BaseTool):
    """Another scan tool for category tests."""

    name = "sample_tool_scan"
    description = "Another scan tool"
    category = ToolCategory.SCAN
    parameters = []

    async def execute(self, **kwargs: object) -> ToolResult:
        return success_result({})


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clean registry before each test."""
    get_tool_registry().clear()
    yield  # type: ignore[misc]
    get_tool_registry().clear()


# -----------------------------------------------------------------------------
# Registry Singleton Tests
# -----------------------------------------------------------------------------


class TestRegistrySingleton:
    """Tests for singleton pattern."""

    def test_singleton_instance(self) -> None:
        """Registry should be a singleton."""
        r1 = ToolRegistry()
        r2 = ToolRegistry()
        assert r1 is r2

    def test_get_tool_registry_singleton(self) -> None:
        """get_tool_registry should return the singleton."""
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2


# -----------------------------------------------------------------------------
# Registration Tests
# -----------------------------------------------------------------------------


class TestRegistration:
    """Tests for tool registration."""

    def test_register_and_get(self) -> None:
        """Should register and retrieve tools."""
        registry = get_tool_registry()
        tool = SampleToolA()
        registry.register(tool)

        retrieved = registry.get("sample_tool_a")
        assert retrieved is tool

    def test_register_class(self) -> None:
        """Should register tool by class."""
        registry = get_tool_registry()
        registry.register_class(SampleToolA)

        tool = registry.get("sample_tool_a")
        assert tool is not None
        assert isinstance(tool, SampleToolA)

    def test_duplicate_registration_raises(self) -> None:
        """Should raise on duplicate registration."""
        registry = get_tool_registry()
        registry.register(SampleToolA())

        with pytest.raises(ValueError, match="already registered"):
            registry.register(SampleToolA())

    def test_unregister(self) -> None:
        """Should unregister tools."""
        registry = get_tool_registry()
        registry.register(SampleToolA())

        assert registry.unregister("sample_tool_a") is True
        assert registry.get("sample_tool_a") is None

    def test_unregister_nonexistent(self) -> None:
        """Unregistering nonexistent tool returns False."""
        registry = get_tool_registry()
        assert registry.unregister("nonexistent") is False


# -----------------------------------------------------------------------------
# Lookup Tests
# -----------------------------------------------------------------------------


class TestLookup:
    """Tests for tool lookup."""

    def test_get_nonexistent(self) -> None:
        """Getting nonexistent tool returns None."""
        registry = get_tool_registry()
        assert registry.get("nonexistent") is None

    def test_get_all(self) -> None:
        """Should return all registered tools."""
        registry = get_tool_registry()
        registry.register(SampleToolA())
        registry.register(SampleToolB())

        tools = registry.get_all()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "sample_tool_a" in names
        assert "sample_tool_b" in names

    def test_get_by_category(self) -> None:
        """Should filter tools by category."""
        registry = get_tool_registry()
        registry.register(SampleToolA())  # SCAN
        registry.register(SampleToolB())  # DETECTION
        registry.register(SampleToolScan())  # SCAN

        scan_tools = registry.get_by_category(ToolCategory.SCAN)
        assert len(scan_tools) == 2

        detection_tools = registry.get_by_category(ToolCategory.DETECTION)
        assert len(detection_tools) == 1

    def test_get_names(self) -> None:
        """Should return all tool names."""
        registry = get_tool_registry()
        registry.register(SampleToolA())
        registry.register(SampleToolB())

        names = registry.get_names()
        assert "sample_tool_a" in names
        assert "sample_tool_b" in names

    def test_has(self) -> None:
        """has() should check if tool exists."""
        registry = get_tool_registry()
        registry.register(SampleToolA())

        assert registry.has("sample_tool_a") is True
        assert registry.has("nonexistent") is False

    def test_contains(self) -> None:
        """'in' operator should work."""
        registry = get_tool_registry()
        registry.register(SampleToolA())

        assert "sample_tool_a" in registry
        assert "nonexistent" not in registry

    def test_len(self) -> None:
        """len() should return tool count."""
        registry = get_tool_registry()
        assert len(registry) == 0

        registry.register(SampleToolA())
        assert len(registry) == 1

        registry.register(SampleToolB())
        assert len(registry) == 2


# -----------------------------------------------------------------------------
# Schema Tests
# -----------------------------------------------------------------------------


class TestSchemas:
    """Tests for schema generation."""

    def test_get_schemas(self) -> None:
        """Should return schemas for all tools."""
        registry = get_tool_registry()
        registry.register(SampleToolA())
        registry.register(SampleToolB())

        schemas = registry.get_schemas()
        assert len(schemas) == 2

        names = [s["function"]["name"] for s in schemas]
        assert "sample_tool_a" in names
        assert "sample_tool_b" in names

    def test_get_schemas_empty(self) -> None:
        """Empty registry should return empty list."""
        registry = get_tool_registry()
        schemas = registry.get_schemas()
        assert schemas == []


# -----------------------------------------------------------------------------
# Decorator Tests
# -----------------------------------------------------------------------------


class TestRegisterToolDecorator:
    """Tests for @register_tool decorator."""

    def test_decorator_registers(self) -> None:
        """Decorator should register tool class."""
        registry = get_tool_registry()
        registry.clear()

        @register_tool
        class DecoratedTool(BaseTool):
            name = "decorated_tool"
            description = "A decorated tool"
            category = ToolCategory.UTILITY
            parameters = []

            async def execute(self, **kwargs: object) -> ToolResult:
                return success_result({})

        assert registry.get("decorated_tool") is not None

    def test_decorator_returns_class(self) -> None:
        """Decorator should return the original class."""
        registry = get_tool_registry()
        registry.clear()

        @register_tool
        class AnotherDecoratedTool(BaseTool):
            name = "another_decorated"
            description = "Test"
            category = ToolCategory.UTILITY
            parameters = []

            async def execute(self, **kwargs: object) -> ToolResult:
                return success_result({})

        assert AnotherDecoratedTool.name == "another_decorated"
        # Can still instantiate
        instance = AnotherDecoratedTool()
        assert instance.name == "another_decorated"


# -----------------------------------------------------------------------------
# Clear Tests
# -----------------------------------------------------------------------------


class TestClear:
    """Tests for clearing the registry."""

    def test_clear(self) -> None:
        """clear() should remove all tools."""
        registry = get_tool_registry()
        registry.register(SampleToolA())
        registry.register(SampleToolB())

        registry.clear()
        assert len(registry) == 0
        assert registry.get("sample_tool_a") is None
