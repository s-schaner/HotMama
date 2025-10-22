"""Configuration for the GUI service."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-driven configuration for the GUI."""

    api_base_url: HttpUrl = Field(
        default="http://api:8000/v1", alias="GUI_API_BASE_URL"
    )
    request_timeout: float = Field(default=15.0, alias="GUI_REQUEST_TIMEOUT")
    artifact_root: Path = Field(default=Path("/app/sessions"), alias="ARTIFACT_DIR")
    upload_dir: Path = Field(
        default=Path("/app/sessions/gui/uploads"), alias="GUI_UPLOAD_DIR"
    )
    download_dir: Path = Field(
        default=Path("/app/sessions/gui/downloads"), alias="GUI_DOWNLOAD_DIR"
    )
    poll_interval: float = Field(default=2.0, alias="GUI_POLL_INTERVAL")
    host: str = Field(default="0.0.0.0", alias="GUI_HOST")
    port: int = Field(default=7860, alias="GUI_PORT")
    service_name: str = Field(default="hotmama-gui", alias="SERVICE_NAME")
    log_json: bool = Field(default=False, alias="LOG_JSON")

    # LM Studio configuration (local inference)
    lmstudio_base_url: str | None = Field(
        default="http://127.0.0.1:1234", alias="GUI_LMSTUDIO_BASE_URL"
    )
    lmstudio_api_key: str = Field(default="lm-studio", alias="GUI_LMSTUDIO_API_KEY")
    lm_parser_model: str | None = Field(
        default="qwen2.5-3b-instruct", alias="GUI_LM_PARSER_MODEL"
    )
    lm_vision_model: str | None = Field(
        default="qwen/qwen2.5-vl-7b", alias="GUI_LM_VISION_MODEL"
    )
    lm_enrichment_model: str | None = Field(
        default="qwen2.5-vl-7b", alias="GUI_LM_ENRICHMENT_MODEL"
    )
    lm_system_prompt: str | None = Field(
        default=None, alias="GUI_LM_SYSTEM_PROMPT"
    )
    lm_temperature: float = Field(default=0.0, alias="GUI_LM_TEMPERATURE")
    lm_max_tokens: int = Field(default=512, alias="GUI_LM_MAX_TOKENS")

    # Hugging Face configuration (cloud inference)
    huggingface_api_url: str | None = Field(
        default=None, alias="GUI_HUGGINGFACE_API_URL"
    )
    huggingface_api_key: str | None = Field(
        default=None, alias="GUI_HUGGINGFACE_API_KEY"
    )
    huggingface_model: str | None = Field(
        default=None, alias="GUI_HUGGINGFACE_MODEL"
    )

    # LLM provider selection: "lmstudio" or "huggingface"
    llm_provider: str = Field(default="lmstudio", alias="GUI_LLM_PROVIDER")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.download_dir.mkdir(parents=True, exist_ok=True)
    return settings
