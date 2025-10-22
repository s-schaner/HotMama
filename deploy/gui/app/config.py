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
