"""Centralized application settings using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="VOLLEYSENSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Host interface to bind")
    port: int = Field(default=7860, description="Port to serve on")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Directory settings
    root_dir: Path = Field(default_factory=lambda: Path.cwd(), description="Application root directory")
    sessions_dir: Path = Field(default_factory=lambda: Path.cwd() / "sessions", description="Sessions storage directory")
    log_dir: Path | None = Field(default=None, description="Log directory (defaults to sessions_dir/logs)")

    # Database settings
    db_url: str | None = Field(default=None, description="Database URL (defaults to SQLite in sessions_dir)")

    # Upload limits
    max_file_size_mb: int = Field(default=500, description="Maximum upload file size in MB")
    allowed_video_extensions: list[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"],
        description="Allowed video file extensions",
    )

    # Video processing defaults
    default_fps: int = Field(default=3, ge=1, le=60, description="Default frame extraction FPS")
    default_max_frames: int = Field(default=48, ge=1, le=500, description="Default maximum frames to extract")
    default_jpeg_quality: int = Field(default=85, ge=1, le=100, description="Default JPEG quality for frame encoding")

    # LLM settings
    default_llm_endpoint: str = Field(
        default="https://api-inference.huggingface.co",
        description="Default LLM endpoint URL",
    )
    default_llm_model: str = Field(
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        description="Default LLM model",
    )
    llm_timeout_seconds: int = Field(default=120, ge=10, description="LLM API timeout in seconds")
    llm_max_tokens: int = Field(default=256, ge=1, description="Maximum LLM response tokens")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM temperature")

    # API security
    auth_token: str | None = Field(default=None, description="Optional API authentication token")
    enable_cors: bool = Field(default=False, description="Enable CORS")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")

    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable API rate limiting")
    rate_limit_per_minute: int = Field(default=10, ge=1, description="API requests per minute per IP")

    # Performance
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache TTL in seconds")

    def get_db_url(self) -> str:
        """Get the database URL, with fallback to default SQLite location."""
        if self.db_url:
            return self.db_url
        db_path = self.sessions_dir / "volleysense.db"
        return f"sqlite:///{db_path}"

    def get_log_dir(self) -> Path:
        """Get the log directory, with fallback to sessions_dir/logs."""
        if self.log_dir:
            return self.log_dir
        return self.sessions_dir / "logs"

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.get_log_dir().mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)."""
    global _settings
    _settings = None
    return get_settings()
