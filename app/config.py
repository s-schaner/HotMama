from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class AppConfig:
    """Configuration flags and directories for the VolleySense app."""

    root_dir: Path = field(default_factory=lambda: Path(os.environ.get("VOLLEYSENSE_ROOT", Path.cwd())))
    data_dir: Path = field(init=False)
    sessions_dir: Path = field(init=False)
    db_url: str = field(init=False)
    hf_endpoint: str | None = field(default_factory=lambda: os.environ.get("HF_OPENAI_ENDPOINT"))
    llm_model: str = field(default_factory=lambda: os.environ.get("VOLLEYSENSE_LLM_MODEL", "gpt-4o-mini"))
    auth_token: str | None = field(default_factory=lambda: os.environ.get("VOLLEYSENSE_AUTH_TOKEN"))
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data"
        self.sessions_dir = self.root_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.sessions_dir / "volleysense.db"
        self.db_url = f"sqlite:///{db_path}"


def load_config() -> AppConfig:
    """Load configuration from environment variables and defaults."""

    cfg = AppConfig()
    if cfg.hf_endpoint:
        cfg.extra["hf_endpoint"] = cfg.hf_endpoint
    return cfg
