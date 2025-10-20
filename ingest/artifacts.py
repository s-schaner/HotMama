from __future__ import annotations

from pathlib import Path


def session_artifact_dir(base: Path, session_id: str) -> Path:
    path = base / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_debug_text(path: Path, name: str, content: str) -> Path:
    target = path / name
    target.write_text(content, encoding="utf-8")
    return target
