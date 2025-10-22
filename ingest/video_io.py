from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable


class VideoResolver:
    """Simplified video resolver handling path dictionaries and strings."""

    def resolve(self, source: str | Dict[str, Any]) -> Path:
        if isinstance(source, dict):
            for key in ("path", "name", "video", "filename", "tempfile"):
                if key in source and source[key]:
                    return Path(source[key]).expanduser().resolve()
            raise ValueError("Video dictionary missing path-like keys")
        return Path(source).expanduser().resolve()

    def sample_frames(
        self, source: str | Dict[str, Any], step: int = 30
    ) -> Iterable[int]:
        path = self.resolve(source)
        size = path.stat().st_size if path.exists() else 100
        total_frames = max(1, size // 1024)
        return range(0, total_frames, step)
