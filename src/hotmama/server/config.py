"""Server settings — everything overridable via HOTMAMA_* env vars."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HOTMAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = Path("data/hotmama.db")
    ui_dist: Path | None = None
    """Path to the built PWA (ui/dist). Auto-detected relative to the repo if unset."""

    media_root: Path = Path("data/media")
    segment_seconds: float = 60.0
    """Recording segment length; also the worst-case clip latency."""
    tag_pre_seconds: float = 8.0
    tag_post_seconds: float = 4.0
    rally_pad_seconds: float = 2.0
    rally_max_seconds: float = 60.0

    worker_token: str | None = None
    """Bearer token for the remote-worker feed. Unset = feed disabled (default)."""
    worker_lease_seconds: float = 300.0
    """How long a leased analysis job stays claimed before re-queueing."""

    auto_commit_confidence: float | None = None
    """Auto-commit CV proposals at/above this confidence. None (default) =
    never auto-commit — every proposal waits for a human. Manual-first."""

    def resolve_ui_dist(self) -> Path | None:
        if self.ui_dist is not None:
            return self.ui_dist if self.ui_dist.is_dir() else None
        candidate = Path(__file__).resolve().parents[3] / "ui" / "dist"
        return candidate if candidate.is_dir() else None
