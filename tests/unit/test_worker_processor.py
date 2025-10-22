from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "deploy" / "worker-vision"))

from app.config import Settings  # noqa: E402
from app.processor import VisionProcessor  # noqa: E402
from app.schemas import QueuedJob  # noqa: E402


def test_processor_creates_artifact(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path)
    processor = VisionProcessor(settings)
    job = QueuedJob(
        job_id=uuid4(),
        submitted_at=datetime.utcnow(),
        status="queued",
        profile="cpu",
        payload={"source_uri": ""},
    )

    result = processor.process(job)

    assert result.status == "completed"
    assert result.artifact_path is not None
    assert tmp_path.joinpath(str(job.job_id), "result.json").exists()
