"""End-to-end capture flow through the API: record → tag → rally → clips ready."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("cv2")

from hotmama.server.app import create_app  # noqa: E402
from hotmama.server.config import Settings  # noqa: E402

from ..capture.util import make_dummy_video  # noqa: E402
from .test_api import LINEUP, ROSTER  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        db_path=tmp_path / "test.db",
        media_root=tmp_path / "media",
        ui_dist=None,
        segment_seconds=1.0,
        tag_pre_seconds=2.0,
        tag_post_seconds=0.5,
        rally_pad_seconds=0.3,
        rally_max_seconds=5.0,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _session_with_set(client: TestClient) -> str:
    response = client.post(
        "/api/sessions",
        json={"our_team": "HotMama", "opponent": "Rivals", "roster": ROSTER},
    )
    assert response.status_code == 201
    session_id = str(response.json()["session_id"])
    response = client.post(
        f"/api/sessions/{session_id}/events",
        json={
            "event": {
                "type": "set_started",
                "set_number": 1,
                "lineup": LINEUP,
                "liberos": ["p8"],
            }
        },
    )
    assert response.status_code == 200
    return session_id


def _poll(check: Any, timeout: float, message: str) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result is not None:
            return result
        time.sleep(0.3)
    pytest.fail(f"timed out: {message}")


def test_capture_records_and_clips(client: TestClient, tmp_path: Path) -> None:
    session_id = _session_with_set(client)
    video = make_dummy_video(tmp_path / "feed.mp4", seconds=3.0, fps=20)

    response = client.post(
        f"/api/sessions/{session_id}/capture", json={"source": str(video)}
    )
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "recording"

    # A second start while recording is a conflict.
    conflict = client.post(
        f"/api/sessions/{session_id}/capture", json={"source": str(video)}
    )
    assert conflict.status_code == 409

    time.sleep(1.0)  # let the recording establish some footage
    tagged = client.post(
        f"/api/sessions/{session_id}/events",
        json={"event": {"type": "moment_tagged", "tag": "highlight"}},
    )
    assert tagged.status_code == 200
    rally = client.post(
        f"/api/sessions/{session_id}/events",
        json={"event": {"type": "rally_ended", "winner": "us", "reason": "kill"}},
    )
    assert rally.status_code == 200

    _poll(
        lambda: (
            True
            if client.get(f"/api/sessions/{session_id}/capture").json()["state"]
            == "finished"
            else None
        ),
        timeout=20.0,
        message="recording did not finish",
    )

    def clips_ready() -> list[dict[str, Any]] | None:
        clips = client.get(f"/api/sessions/{session_id}/clips").json()
        if len(clips) >= 2 and all(c["status"] in ("ready", "failed") for c in clips):
            return list(clips)
        return None

    clips = _poll(clips_ready, timeout=30.0, message="clips never resolved")
    by_kind = {clip["kind"]: clip for clip in clips}
    assert set(by_kind) == {"tag", "rally"}
    for clip in clips:
        assert clip["status"] == "ready", clip
        media = client.get(clip["url"])
        assert media.status_code == 200
        assert len(media.content) > 0

    tag_clip = by_kind["tag"]
    assert tag_clip["label"] == "highlight"
    assert tag_clip["url"].endswith(".mp4")


def test_capture_bad_source_is_400(client: TestClient, tmp_path: Path) -> None:
    session_id = _session_with_set(client)
    response = client.post(
        f"/api/sessions/{session_id}/capture",
        json={"source": str(tmp_path / "missing.mp4")},
    )
    assert response.status_code == 400


def test_capture_status_idle_by_default(client: TestClient) -> None:
    session_id = _session_with_set(client)
    response = client.get(f"/api/sessions/{session_id}/capture")
    assert response.json() == {"state": "idle"}
    assert client.get(f"/api/sessions/{session_id}/clips").json() == []


def test_capture_unknown_session_404(client: TestClient) -> None:
    assert client.get("/api/sessions/s_nope/capture").status_code == 404
    assert (
        client.post("/api/sessions/s_nope/capture", json={"source": "0"}).status_code
        == 404
    )
