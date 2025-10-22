from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from deploy.gui.app.client import JobHandle, JobState
from deploy.gui.app.controller import GuiController
from deploy.gui.app.storage import StorageManager


class FakeApiClient:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, dict, str]] = []
        self.state: JobState | None = None
        self.last_job_type: str | None = None

    def submit_job(
        self,
        source_uri: str,
        parameters: dict | None = None,
        job_type: str = "vision.process",
    ) -> JobHandle:
        job_id = uuid4()
        handle = JobHandle(
            job_id=job_id,
            status="queued",
            submitted_at=datetime.utcnow(),
            profile="cpu",
        )
        self.submissions.append((source_uri, parameters or {}, job_type))
        self.last_job_type = job_type
        self.state = JobState(
            job_id=job_id,
            status="queued",
            submitted_at=handle.submitted_at,
            updated_at=handle.submitted_at,
            profile="cpu",
            message=None,
            artifact_path=None,
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

    job_id, message = gui.submit_job(str(video), "{}")

    assert UUID(job_id)
    assert "queued" in message
    assert client.submissions


def test_submit_job_with_custom_type(
    controller: tuple[GuiController, FakeApiClient], tmp_path
) -> None:
    gui, client = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, _ = gui.submit_job(str(video), "{}", job_type="vision.segment")

    assert UUID(job_id)
    assert client.last_job_type == "vision.segment"


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
    artifact = tmp_path / "result.json"
    artifact.write_text("{}")
    client.state = JobState(
        job_id=UUID(job_id),
        status="completed",
        submitted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        profile="cpu",
        message=None,
        artifact_path=str(artifact),
    )

    payload, message, artifact_path = gui.refresh_status(job_id)

    assert payload["status"] == "completed"
    assert message.startswith("✅")
    assert artifact_path.endswith("result.json")


def test_refresh_status_invalid_id(controller: tuple[GuiController, FakeApiClient]) -> None:
    gui, _ = controller
    payload, message, artifact_path = gui.refresh_status("not-a-uuid")

    assert payload == {}
    assert artifact_path == ""
    assert message.startswith("❌")
