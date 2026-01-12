"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CLOUMASK_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8765
    debug: bool = False

    # Ollama/LLM settings (for 12-dev-workflow-ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    llm_cache_size: int = 4  # Max cached LLM instances

    # Model settings (for 03-cv-models)
    models_dir: str = "models"
    device: str = "cuda"  # or "cpu", "mps"


# Global settings instance
settings = Settings()
