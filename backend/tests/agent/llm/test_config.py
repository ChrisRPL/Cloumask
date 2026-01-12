"""Tests for LLM configuration module."""


from backend.agent.llm.config import (
    LLM_CONFIGS,
    LLMConfig,
    LLMUseCase,
    get_config_for_use_case,
    get_default_config,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = LLMConfig(model="test-model")

        assert config.model == "test-model"
        assert config.temperature == 0.1
        assert config.max_tokens == 4096
        assert config.timeout == 120
        assert config.base_url == "http://localhost:11434"
        assert config.num_ctx == 8192
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.retry_backoff == 2.0
        assert config.fallback_models == []

    def test_custom_values(self):
        """Config should accept custom values."""
        config = LLMConfig(
            model="custom-model",
            temperature=0.5,
            max_tokens=2048,
            timeout=60,
            base_url="http://custom:11434",
            num_ctx=4096,
            max_retries=5,
            retry_delay=0.5,
            retry_backoff=3.0,
            fallback_models=["fallback1", "fallback2"],
        )

        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.timeout == 60
        assert config.base_url == "http://custom:11434"
        assert config.num_ctx == 4096
        assert config.max_retries == 5
        assert config.retry_delay == 0.5
        assert config.retry_backoff == 3.0
        assert config.fallback_models == ["fallback1", "fallback2"]

    def test_with_temperature(self):
        """with_temperature should return a new config."""
        config = LLMConfig(model="test", temperature=0.1)
        new_config = config.with_temperature(0.7)

        assert config.temperature == 0.1  # Original unchanged
        assert new_config.temperature == 0.7
        assert new_config.model == config.model

    def test_with_model(self):
        """with_model should return a new config."""
        config = LLMConfig(model="model1")
        new_config = config.with_model("model2")

        assert config.model == "model1"  # Original unchanged
        assert new_config.model == "model2"
        assert new_config.temperature == config.temperature


class TestLLMUseCase:
    """Tests for LLMUseCase enum."""

    def test_use_cases_exist(self):
        """All expected use cases should be defined."""
        assert LLMUseCase.TOOL_CALLING == "tool_calling"
        assert LLMUseCase.CONVERSATION == "conversation"
        assert LLMUseCase.PLANNING == "planning"
        assert LLMUseCase.JSON_OUTPUT == "json_output"

    def test_use_cases_are_strings(self):
        """Use cases should be string enums."""
        for use_case in LLMUseCase:
            assert isinstance(use_case.value, str)


class TestLLMConfigs:
    """Tests for default configuration presets."""

    def test_all_use_cases_have_configs(self):
        """Each use case should have a default config."""
        for use_case in LLMUseCase:
            assert use_case in LLM_CONFIGS
            assert isinstance(LLM_CONFIGS[use_case], LLMConfig)

    def test_tool_calling_config(self):
        """Tool calling should have low temperature."""
        config = LLM_CONFIGS[LLMUseCase.TOOL_CALLING]

        assert config.temperature == 0.1
        assert config.max_tokens == 2048
        assert "qwen3:8b" in config.fallback_models

    def test_conversation_config(self):
        """Conversation should have higher temperature."""
        config = LLM_CONFIGS[LLMUseCase.CONVERSATION]

        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_planning_config(self):
        """Planning should have moderate temperature."""
        config = LLM_CONFIGS[LLMUseCase.PLANNING]

        assert config.temperature == 0.3
        assert config.max_tokens == 4096

    def test_json_output_config(self):
        """JSON output should have zero temperature."""
        config = LLM_CONFIGS[LLMUseCase.JSON_OUTPUT]

        assert config.temperature == 0.0
        assert config.max_tokens == 2048

    def test_temperature_ordering(self):
        """Temperature should increase: JSON < Tool < Planning < Conversation."""
        json_temp = LLM_CONFIGS[LLMUseCase.JSON_OUTPUT].temperature
        tool_temp = LLM_CONFIGS[LLMUseCase.TOOL_CALLING].temperature
        plan_temp = LLM_CONFIGS[LLMUseCase.PLANNING].temperature
        conv_temp = LLM_CONFIGS[LLMUseCase.CONVERSATION].temperature

        assert json_temp <= tool_temp <= plan_temp <= conv_temp


class TestGetConfigForUseCase:
    """Tests for get_config_for_use_case function."""

    def test_returns_correct_config(self):
        """Should return the config for the specified use case."""
        config = get_config_for_use_case(LLMUseCase.TOOL_CALLING)
        assert config == LLM_CONFIGS[LLMUseCase.TOOL_CALLING]

    def test_all_use_cases(self):
        """Should work for all use cases."""
        for use_case in LLMUseCase:
            config = get_config_for_use_case(use_case)
            assert isinstance(config, LLMConfig)


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_tool_calling_config(self):
        """Default should be tool calling config."""
        config = get_default_config()
        assert config == LLM_CONFIGS[LLMUseCase.TOOL_CALLING]

    def test_returns_llm_config(self):
        """Should return an LLMConfig instance."""
        config = get_default_config()
        assert isinstance(config, LLMConfig)
