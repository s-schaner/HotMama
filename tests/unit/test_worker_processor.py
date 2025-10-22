from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "deploy" / "worker-vision"))

from app import processor as processor_module  # noqa: E402
from app.config import Settings  # noqa: E402
from app.processor import VisionProcessor  # noqa: E402
from app.schemas import QueuedJob  # noqa: E402


def _create_dummy_video(path: Path, *, frame_count: int = 8, fps: int = 12) -> None:
    width, height = 64, 48
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("failed to create dummy video")
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = ((index * 25) % 255, (index * 40) % 255, (index * 55) % 255)
        cv2.putText(
            frame,
            f"{index}",
            (5, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _make_job(tmp_path: Path, video: Path) -> QueuedJob:
    return QueuedJob(
        job_id=uuid4(),
        submitted_at=datetime.utcnow(),
        status="queued",
        profile="cpu",
        payload={
            "source_uri": str(video),
            "fps": 12,
            "options": {"overlays": ["heatmap", "tracking", "pose"]},
        },
    )


def test_processor_creates_video_artifact_with_overlays(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path)
    processor = VisionProcessor(settings)
    video = tmp_path / "input.avi"
    _create_dummy_video(video)
    job = _make_job(tmp_path, video)

    result = processor.process(job)

    assert result.status == "completed"
    assert result.artifact_path is not None
    artifact_video = Path(result.artifact_path)
    assert artifact_video.exists()
    assert artifact_video.suffix in {".mp4", ".mkv"}

    metadata_path = tmp_path.joinpath(str(job.job_id), "result.json")
    data = json.loads(metadata_path.read_text())
    assert data["video"]["frames"] == 8
    assert set(data["video"]["overlays_rendered"]) == {
        "heatmap",
        "tracking",
        "pose",
    }
    assert data["video"]["path"] == str(artifact_video)


def test_processor_falls_back_to_mkv_when_mp4_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(artifact_root=tmp_path)
    processor = VisionProcessor(settings)
    video = tmp_path / "input.avi"
    _create_dummy_video(video)
    job = _make_job(tmp_path, video)

    class StubWriter:
        def __init__(self, filename: str, *_: object) -> None:
            self.filename = filename
            self._opened = filename.endswith(".mkv")
            self._written = False

        def isOpened(self) -> bool:
            return self._opened

        def write(self, _frame: np.ndarray) -> None:
            if not self._opened:
                return
            Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
            with open(self.filename, "ab") as handle:
                handle.write(b"frame")
            self._written = True

        def release(self) -> None:
            if self._opened and not Path(self.filename).exists():
                Path(self.filename).write_bytes(b"" if not self._written else b"frame")

    def fake_writer(filename: str, *args: object, **kwargs: object) -> StubWriter:
        return StubWriter(filename, *args, **kwargs)

    monkeypatch.setattr(processor_module.cv2, "VideoWriter", fake_writer)

    result = processor.process(job)

    assert result.status == "completed"
    assert result.artifact_path is not None
    artifact_video = Path(result.artifact_path)
    assert artifact_video.suffix == ".mkv"

    metadata_path = tmp_path.joinpath(str(job.job_id), "result.json")
    data = json.loads(metadata_path.read_text())
    assert data["video"]["fallback_used"] is True
    assert any(attempt["path"].endswith("result.mp4") for attempt in data["video"]["attempts"])


def test_processor_reports_failure_when_all_writers_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(artifact_root=tmp_path)
    processor = VisionProcessor(settings)
    video = tmp_path / "input.avi"
    _create_dummy_video(video)
    job = _make_job(tmp_path, video)

    class ClosedWriter:
        def __init__(self, filename: str, *_: object) -> None:
            self.filename = filename

        def isOpened(self) -> bool:
            return False

        def write(self, _frame: np.ndarray) -> None:  # pragma: no cover - never called
            raise AssertionError("writer should not be used")

        def release(self) -> None:
            Path(self.filename).unlink(missing_ok=True)

    monkeypatch.setattr(processor_module.cv2, "VideoWriter", ClosedWriter)

    result = processor.process(job)

    assert result.status == "failed"
    assert result.artifact_path is None
    assert "video writer" in result.message

    metadata_path = tmp_path.joinpath(str(job.job_id), "result.json")
    data = json.loads(metadata_path.read_text())
    assert data["video"]["error"] == "unable to initialise video writer"
