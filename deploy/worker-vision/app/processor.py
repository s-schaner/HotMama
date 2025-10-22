"""Vision processing pipeline stub."""

from __future__ import annotations

import json
import logging
from typing import Any

import cv2  # type: ignore
import torch

from .config import Settings, get_settings
from .schemas import ProcessResult, QueuedJob

LOGGER = logging.getLogger("hotmama.worker.processor")


class VisionProcessor:
    """Performs lightweight processing to validate the pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: torch.nn.Module | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            LOGGER.info(
                "loading vision model stub",
                extra={"service": self._settings.service_name},
            )
            self._model = torch.nn.Sequential(torch.nn.Identity())

    def _summarize_video(self, source_uri: str) -> dict[str, Any]:
        capture = cv2.VideoCapture(source_uri)
        if not capture.isOpened():
            return {"frames": 0, "fps": 0.0, "valid": False}
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
        return {"frames": frame_count, "fps": fps, "valid": True}

    def process(self, job: QueuedJob) -> ProcessResult:
        self._ensure_model()
        payload = job.payload
        source_uri = payload.get("source_uri", "")
        summary = self._summarize_video(source_uri) if source_uri else {}

        artifact_dir = self._settings.artifact_root / str(job.job_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "result.json"

        tensor = torch.tensor([len(source_uri)], dtype=torch.float32)
        _ = self._model(tensor)
        stats = {
            "job_id": str(job.job_id),
            "profile": job.profile,
            "summary": summary,
            "parameters": payload.get("options")
            if isinstance(payload.get("options"), dict)
            else payload.get("parameters", {}),
        }
        artifact_path.write_text(json.dumps(stats, indent=2))
        return ProcessResult(
            status="completed",
            message="processing complete",
            artifact_path=str(artifact_path),
        )
