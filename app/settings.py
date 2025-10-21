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

    # Player tracking settings
    yolo_model: str = Field(default="yolov8n", description="YOLO model to use (yolov8n/s/m/l/x)")
    detection_confidence: float = Field(default=0.25, ge=0.0, le=1.0, description="YOLO detection confidence threshold")
    detection_iou: float = Field(default=0.45, ge=0.0, le=1.0, description="YOLO IoU threshold for NMS")
    detection_device: str | None = Field(default=None, description="Device for detection (cpu/cuda/cuda:0, auto if None)")

    # Tracking settings
    tracking_max_age: int = Field(default=30, ge=1, description="Max frames to keep track alive without detections")
    tracking_min_hits: int = Field(default=3, ge=1, description="Min detections before track is confirmed")
    tracking_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Min IoU for track-detection association")

    # Pose estimation settings
    enable_pose_estimation: bool = Field(default=True, description="Enable pose estimation")
    pose_model_type: str = Field(default="mediapipe", description="Pose model type (mediapipe/yolov8-pose)")
    pose_model_complexity: int = Field(default=1, ge=0, le=2, description="Pose model complexity (0=lite, 1=full, 2=heavy)")
    pose_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Min confidence for pose detection")

    # Overlay settings
    enable_video_overlays: bool = Field(default=True, description="Enable video overlays by default")
    overlay_draw_boxes: bool = Field(default=True, description="Draw bounding boxes")
    overlay_draw_pose: bool = Field(default=True, description="Draw pose skeletons")
    overlay_draw_trails: bool = Field(default=True, description="Draw movement trails")
    overlay_trail_length: int = Field(default=30, ge=5, le=100, description="Trail length in frames")

    # Analytics settings
    enable_player_analytics: bool = Field(default=True, description="Enable player analytics")
    enable_heatmap_generation: bool = Field(default=True, description="Enable heatmap generation")
    heatmap_kernel_size: int = Field(default=25, ge=5, le=100, description="Gaussian kernel size for heatmap smoothing")
    heatmap_decay_factor: float = Field(default=0.95, ge=0.5, le=1.0, description="Temporal decay factor for heatmap")

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
