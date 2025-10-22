from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

pytest.importorskip("pydantic")

from deploy.api.app.schemas import JobSpec
from deploy.gui.app.client import ApiClient, JobHandle, JobState
from deploy.gui.app.controller import GuiController
from deploy.gui.app.storage import StorageManager


class FakeApiClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, Any]] = []
        self.state: JobState | None = None
        self.last_task: str | None = None
        self.last_manifest: dict[str, Any] | None = None

    def submit_job(
        self,
        source_uri: str,
        parameters: dict | None = None,
        job_type: str = "pipeline.analysis",
        overlays: Sequence[str] | None = None,
        priority: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> JobHandle:
        if manifest is None:
            raise AssertionError("manifest is required")
        job_id = uuid4()
        payload = manifest.get("payload", {})
        options = payload.get("options", {})
        task = payload.get("task", ApiClient._normalise_task(job_type))
        overlay_list = list(options.get("overlays", []))
        handle = JobHandle(
            job_id=job_id,
            status="queued",
            submitted_at=datetime.utcnow(),
            profile="cpu",
            priority=priority or "normal",
            idempotency_key="test",
            task=task,
        )
        record = {
            "source_uri": source_uri,
            "manifest": manifest,
            "overlays": overlay_list,
            "task": task,
        }
        self.submissions.append(record)
        self.last_task = task
        self.last_manifest = manifest
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


class FakeLMStudioClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None

    def generate_manifest(
        self,
        prompt: str,
        *,
        source_uri: str,
        task_hint: str | None,
        enrich: bool,
    ) -> JobSpec:
        self.calls.append(
            {
                "prompt": prompt,
                "source_uri": source_uri,
                "task_hint": task_hint,
                "enrich": enrich,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        base = {
            "payload": {
                "task": task_hint or "analyze_video",
                "source_uri": source_uri,
                "options": {"overlays": ["heatmap"]},
            },
            "priority": "normal",
        }
        return JobSpec.model_validate(base)

    def close(self) -> None:  # pragma: no cover - no-op for tests
        pass


@pytest.fixture
def controller_with_lm(
    tmp_path,
) -> tuple[GuiController, FakeApiClient, FakeLMStudioClient]:
    upload_dir = tmp_path / "uploads"
    download_dir = tmp_path / "downloads"
    storage = StorageManager(upload_dir, download_dir)
    client = FakeApiClient()
    lm_client = FakeLMStudioClient()
    settings = SimpleNamespace(service_name="test-gui")
    ctrl = GuiController(  # type: ignore[arg-type]
        client=client, storage=storage, settings=settings, lm_client=lm_client
    )
    return ctrl, client, lm_client


def test_submit_job_success(controller: tuple[GuiController, FakeApiClient], tmp_path) -> None:
    gui, client = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, message, manifest = gui.submit_job(
        str(video),
        "{}",
        overlay_modes=["overlay.activity_heatmap"],
    )

    assert UUID(job_id)
    assert "queued" in message
    assert client.submissions
    assert client.submissions[0]["overlays"] == ["heatmap"]
    assert "Task: analyze_video" in message
    assert "Priority: normal" in message
    assert manifest["payload"]["options"]["overlays"] == ["heatmap"]
    assert manifest["payload"]["source_uri"] == client.submissions[0]["manifest"]["payload"]["source_uri"]


def test_submit_job_with_custom_type(
    controller: tuple[GuiController, FakeApiClient], tmp_path
) -> None:
    gui, client = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, _, manifest = gui.submit_job(
        str(video),
        "{}",
        job_type="pipeline.segment",
        overlay_modes=["overlay.object_tracking", "overlay.pose_skeleton"],
    )

    assert UUID(job_id)
    assert client.last_task == "extract_clips"
    assert client.submissions[0]["overlays"] == ["tracking", "pose"]
    assert manifest["payload"]["task"] == "extract_clips"


def test_submit_job_invalid_json(controller: tuple[GuiController, FakeApiClient], tmp_path) -> None:
    gui, _ = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, message, manifest = gui.submit_job(str(video), "{not json}")
    assert job_id == ""
    assert message.startswith("❌")
    assert manifest is None


def test_submit_job_invalid_manifest_structure(
    controller: tuple[GuiController, FakeApiClient], tmp_path
) -> None:
    gui, _ = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, message, manifest = gui.submit_job(str(video), "[]")

    assert job_id == ""
    assert manifest is None
    assert "Parameters JSON must decode" in message


def test_submit_job_natural_language_success(
    controller_with_lm: tuple[GuiController, FakeApiClient, FakeLMStudioClient],
    tmp_path,
) -> None:
    gui, client, lm_client = controller_with_lm
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, message, manifest = gui.submit_job(
        str(video),
        "{}",
        input_mode="nl",
        prompt_text="Analyse the full video",
        overlay_modes=["overlay.pose_skeleton"],
        enrich_manifest=True,
    )

    assert UUID(job_id)
    assert message.startswith("✅")
    assert manifest["payload"]["options"]["overlays"] == ["pose"]
    assert lm_client.calls and lm_client.calls[0]["enrich"] is True
    assert client.submissions[0]["manifest"]["payload"]["task"] == "analyze_video"


def test_submit_job_natural_language_failure(
    controller_with_lm: tuple[GuiController, FakeApiClient, FakeLMStudioClient],
    tmp_path,
) -> None:
    gui, _, lm_client = controller_with_lm
    lm_client.fail_with = RuntimeError("parser offline")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")

    job_id, message, manifest = gui.submit_job(
        str(video),
        "{}",
        input_mode="nl",
        prompt_text="Analyse",
    )

    assert job_id == ""
    assert manifest is None
    assert message.startswith("❌ Failed to parse natural language request")


def test_refresh_status_with_artifact(
    controller: tuple[GuiController, FakeApiClient], tmp_path
) -> None:
    gui, client = controller
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    job_id, _, _ = gui.submit_job(str(video), "{}")
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
