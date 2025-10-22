"""Filesystem helpers for the GUI service."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


class StorageManager:
    """Manage persistent storage for uploaded assets and artifacts."""

    def __init__(self, upload_dir: Path, download_dir: Path) -> None:
        self.upload_dir = upload_dir
        self.download_dir = download_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_path: str) -> Path:
        """Persist an uploaded file into the shared sessions directory."""

        source = Path(file_path)
        suffix = source.suffix
        target = self.upload_dir / f"{uuid.uuid4().hex}{suffix}"
        shutil.copy2(source, target)
        return target

    def prepare_artifact(self, artifact_path: str) -> Path:
        """Copy an artifact into the download directory for exposure via Gradio."""

        source = Path(artifact_path)
        if not source.exists():
            raise FileNotFoundError(f"artifact not found: {artifact_path}")
        destination = self.download_dir / source.name
        if source.resolve() == destination.resolve():
            return destination
        shutil.copy2(source, destination)
        return destination


__all__ = ["StorageManager"]
