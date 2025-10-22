from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

pytest.importorskip("pydantic")

from deploy.gui.app.client import ApiClient, JobHandle, JobState
from deploy.gui.app.controller import GuiController
from deploy.gui.app.storage import StorageManager


class FakeApiClient:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, dict, list[str], str]] = []
        self.state: JobState | None = None
        self.last_task: str | None = None

    def submit_job(
        self,
        source_uri: str,
        parameters: dict | None = None,
        job_type: str = "pipeline.analysis",
        overlays: Sequence[str] | None = None,
        priority: str | None = None,
    ) -> JobHandle:
        job_id = uuid4()
        task = ApiClient._normalise_task(job_type)
        overlay_list = ApiClient._normalise_overlay_modes(overlays) or []
        handle = JobHandle(
            job_id=job_id,
            status="queued",
            submitted_at=datetime.utcnow(),
            profile="cpu",
            priority=priority or "normal",
            idempotency_key="test",
            task=task,
        )
        self.submissions.append((source_uri, parameters or {}, overlay_list, task))
        self.last_task = task
        self.state = JobState(
            job_id=job_id,
            status="queued",
            submitted_at=handle.submitted_at,
            updated_at=handle.submitted_at,
            profile="cpu",
            message=None,
            artifact_path=None,
            priority=handle.priority,
        )
        return handle

    def get_status(self, job_id: UUID) -> JobState:
        if not self.state or self.state.job_id != job_id:
            raise RuntimeError("unknown job")
        return self.state

    def close(self) -> None:  # pragma: no cover - no-op for tests
        pass


@pytest.fixture
def controller(tmp_path) -> tuple[GuiController, FakeApiClient]:
    upload_dir = tmp_path / "uploads"
    download_dir = tmp_path / "downloads"
    storage = StorageManager(upload_dir, download_dir)
    client = FakeApiClient()
    settings = SimpleNamespace(service_name="test-gui")
    ctrl = GuiController(client=client, storage=storage, settings=settings)  # type: ignore[arg-type]
    return ctrl, client


def test_submit_job_success(controller: tuple[GuiController, FakeApiClient], tmp_path) -> None:
    gui, client = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, message = gui.submit_job(
        str(video),
        "{}",
        overlay_modes=["overlay.activity_heatmap"],
    )

    assert UUID(job_id)
    assert "queued" in message
    assert client.submissions
    assert client.submissions[0][2] == ["heatmap"]
    assert "Task: analyze_video" in message
    assert "Priority: normal" in message


def test_submit_job_with_custom_type(
    controller: tuple[GuiController, FakeApiClient], tmp_path
) -> None:
    gui, client = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, _ = gui.submit_job(
        str(video),
        "{}",
        job_type="pipeline.segment",
        overlay_modes=["overlay.object_tracking", "overlay.pose_skeleton"],
    )

    assert UUID(job_id)
    assert client.last_task == "extract_clips"
    assert client.submissions[0][2] == ["tracking", "pose"]


def test_submit_job_invalid_json(controller: tuple[GuiController, FakeApiClient], tmp_path) -> None:
    gui, _ = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, message = gui.submit_job(str(video), "{not json}")
    assert job_id == ""
    assert message.startswith("❌")


def test_refresh_status_with_artifact(
    controller: tuple[GuiController, FakeApiClient], tmp_path
) -> None:
    gui, client = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    job_id, _ = gui.submit_job(str(video), "{}")
    artifact = tmp_path / "result.mp4"
    artifact.write_bytes(b"video")
    client.state = JobState(
        job_id=UUID(job_id),
        status="completed",
        submitted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        profile="cpu",
        message=None,
        artifact_path=str(artifact),
        priority="high",
    )

    payload, message, artifact_path = gui.refresh_status(job_id)

    assert payload["status"] == "completed"
    assert message.startswith("✅")
    assert artifact_path.endswith(".mp4") or artifact_path.endswith(".mkv")


def test_refresh_status_invalid_id(controller: tuple[GuiController, FakeApiClient]) -> None:
    gui, _ = controller
    payload, message, artifact_path = gui.refresh_status("not-a-uuid")

    assert payload == {}
    assert artifact_path == ""
    assert message.startswith("❌")
