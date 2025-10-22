"""Configuration for the vision worker."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = Field(default="redis://queue:6379/0", alias="REDIS_URL")
    redis_queue_name: str = Field(default="hotmama:jobs", alias="REDIS_QUEUE_NAME")
    redis_status_prefix: str = Field(
        default="hotmama:job", alias="REDIS_STATUS_PREFIX"
    )
    artifact_root: Path = Field(default=Path("/app/sessions"), alias="ARTIFACT_DIR")
    poll_timeout: int = Field(default=5, alias="POLL_TIMEOUT")
    max_backoff_seconds: int = Field(default=30, alias="MAX_BACKOFF_SECONDS")
    service_name: str = Field(default="hotmama-worker", alias="SERVICE_NAME")
    deployment_profile: Literal["cpu", "gpu", "rocm"] = Field(
        default="cpu", alias="PROFILE"
    )
    log_json: bool = Field(default=False, alias="LOG_JSON")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
