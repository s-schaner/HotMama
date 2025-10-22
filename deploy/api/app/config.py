"""Configuration management for the API service."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-driven configuration."""

    redis_url: str = Field(default="redis://queue:6379/0", alias="REDIS_URL")
    redis_queue_name: str = Field(default="hotmama:jobs", alias="REDIS_QUEUE_NAME")
    redis_status_prefix: str = Field(
        default="hotmama:job", alias="REDIS_STATUS_PREFIX"
    )
    artifact_dir: Path = Field(default=Path("/app/sessions"), alias="ARTIFACT_DIR")
    log_json: bool = Field(default=False, alias="LOG_JSON")
    service_name: str = Field(default="hotmama-api", alias="SERVICE_NAME")
    api_prefix: str = Field(default="/v1", alias="API_PREFIX")
    default_poll_timeout: int = Field(default=5, alias="DEFAULT_POLL_TIMEOUT")
    result_ttl_seconds: int = Field(default=3600, alias="RESULT_TTL_SECONDS")
    allowed_origins: list[str] = Field(default_factory=list, alias="ALLOWED_ORIGINS")
    deployment_profile: Literal["cpu", "gpu", "rocm"] = Field(
        default="cpu", alias="PROFILE"
    )
    lmstudio_base_url: str | None = Field(default=None, alias="LMSTUDIO_BASE_URL")
    lmstudio_api_key: str = Field(default="lm-studio", alias="LMSTUDIO_API_KEY")
    lm_parser_model: str | None = Field(default=None, alias="LM_PARSER_MODEL")
    lm_parser_temperature: float = Field(default=0.0, alias="LM_PARSER_TEMPERATURE")
    lm_parser_max_tokens: int = Field(default=512, alias="LM_PARSER_MAX_TOKENS")
    lm_parser_system_prompt: str | None = Field(
        default=None, alias="LM_PARSER_PROMPT"
    )

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

    return Settings()
