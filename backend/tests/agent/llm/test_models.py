"""Tests for model management module."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.agent.llm.models import (
    check_model_supports_tools,
    ensure_model_available,
    get_model_info,
    list_available_models,
)


class TestListAvailableModels:
    """Tests for list_available_models function."""

    @pytest.mark.asyncio
    async def test_parses_ollama_list_output(self):
        """Should parse ollama list output correctly."""
        mock_output = (
            "NAME                  SIZE    MODIFIED\n"
            "qwen3:14b             8.9 GB  2 days ago\n"
            "llama3:8b             4.7 GB  1 week ago\n"
        )

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(mock_output.encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            models = await list_available_models()

        assert models == ["qwen3:14b", "llama3:8b"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        """Should return empty list on subprocess error."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            models = await list_available_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self):
        """Should return empty list on timeout."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=TimeoutError(),
        ):
            models = await list_available_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_ollama_not_found(self):
        """Should return empty list when ollama command not found."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError(),
        ):
            models = await list_available_models()

        assert models == []


class TestEnsureModelAvailable:
    """Tests for ensure_model_available function."""

    @pytest.mark.asyncio
    async def test_returns_true_if_model_exists(self):
        """Should return True if model is already available."""
        with patch(
            "backend.agent.llm.models.list_available_models",
            return_value=["qwen3:14b", "llama3:8b"],
        ):
            result = await ensure_model_available("qwen3:14b")

        assert result is True

    @pytest.mark.asyncio
    async def test_matches_model_without_tag(self):
        """Should match model base name without specific tag."""
        with patch(
            "backend.agent.llm.models.list_available_models",
            return_value=["qwen3:latest"],
        ):
            result = await ensure_model_available("qwen3")

        assert result is True

    @pytest.mark.asyncio
    async def test_pulls_model_if_not_available(self):
        """Should attempt to pull model if not available."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Success", b""))

        with (
            patch(
                "backend.agent.llm.models.list_available_models",
                return_value=[],
            ),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            result = await ensure_model_available("new-model")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_if_pull_fails(self):
        """Should return False if model pull fails."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error"))

        with (
            patch(
                "backend.agent.llm.models.list_available_models",
                return_value=[],
            ),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            result = await ensure_model_available("failing-model")

        assert result is False


class TestGetModelInfo:
    """Tests for get_model_info function."""

    @pytest.mark.asyncio
    async def test_parses_modelfile_output(self):
        """Should parse modelfile output correctly."""
        mock_output = (
            "# Modelfile\n"
            "FROM qwen3:14b\n"
            "PARAMETER temperature 0.7\n"
            "PARAMETER top_p 0.9\n"
        )

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(mock_output.encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            info = await get_model_info("qwen3:14b")

        assert info is not None
        assert info["name"] == "qwen3:14b"
        assert "parameters" in info
        assert info["parameters"]["temperature"] == "0.7"

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        """Should return None on error."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            info = await get_model_info("nonexistent")

        assert info is None


class TestCheckModelSupportsTools:
    """Tests for check_model_supports_tools function."""

    @pytest.mark.asyncio
    async def test_recognizes_known_tool_capable_models(self):
        """Should recognize known tool-capable models."""
        with patch(
            "backend.agent.llm.models.get_model_details",
            return_value=None,
        ):
            # Test various known models
            assert await check_model_supports_tools("qwen3:14b") is True
            assert await check_model_supports_tools("llama3:8b") is True
            assert await check_model_supports_tools("llama4:latest") is True
            assert await check_model_supports_tools("mistral:7b") is True
            assert await check_model_supports_tools("command-r:latest") is True

    @pytest.mark.asyncio
    async def test_checks_model_template_for_tools(self):
        """Should check model template for tool support."""
        with patch(
            "backend.agent.llm.models.get_model_details",
            return_value={"template": "You can use tools to help answer."},
        ):
            result = await check_model_supports_tools("custom-model")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_model(self):
        """Should return False for unknown model without tool support."""
        with patch(
            "backend.agent.llm.models.get_model_details",
            return_value={"template": "Simple chat template"},
        ):
            result = await check_model_supports_tools("unknown-model")

        assert result is False
