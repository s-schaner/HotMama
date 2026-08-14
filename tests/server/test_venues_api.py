"""Venues: one saved calibration per gym, delivered to workers with each job."""

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

TOKEN = "worker-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

CALIBRATION = {
    "image_corners": [[100.0, 900.0], [1800.0, 890.0], [1500.0, 300.0], [400.0, 305.0]],
    "frame_width": 1920,
    "frame_height": 1080,
    "mode": "near_half",
}


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
        ui_dist=None,
        segment_seconds=1.0,
        rally_pad_seconds=0.3,
        rally_max_seconds=5.0,
        worker_token=TOKEN,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


class TestVenueCrud:
    def test_save_list_delete(self, client: TestClient) -> None:
        assert (
            client.put("/api/venues/Main Gym", json={"calibration": CALIBRATION})
        ).status_code == 200
        venues = client.get("/api/venues").json()
        assert venues[0]["name"] == "Main Gym"
        assert venues[0]["calibration"]["mode"] == "near_half"
        assert client.delete("/api/venues/Main Gym").status_code == 200
        assert client.delete("/api/venues/Main Gym").status_code == 404

    def test_invalid_calibration_rejected(self, client: TestClient) -> None:
        bad = {**CALIBRATION, "image_corners": [[0, 0], [1, 1], [2, 2], [3, 3]]}
        response = client.put("/api/venues/bad", json={"calibration": bad})
        assert response.status_code == 422

    def test_session_requires_known_venue(self, client: TestClient) -> None:
        response = client.post("/api/sessions", json={"venue": "nowhere"})
        assert response.status_code == 422


class TestCalibrationReachesWorkers:
    def test_lease_includes_venue_calibration(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        client.put("/api/venues/Main Gym", json={"calibration": CALIBRATION})
        response = client.post(
            "/api/sessions",
            json={"our_team": "HotMama", "roster": ROSTER, "venue": "Main Gym"},
        )
        session_id = response.json()["session_id"]
        client.post(
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
        video = make_dummy_video(tmp_path / "feed.mp4", seconds=2.0, fps=20)
        client.post(
            f"/api/sessions/{session_id}/capture",
            json={"source": str(video), "tag_clips": False},
        )
        time.sleep(0.6)
        client.post(
            f"/api/sessions/{session_id}/events",
            json={"event": {"type": "rally_ended", "winner": "us"}},
        )

        def leased() -> dict[str, Any] | None:
            response = client.post(
                "/api/worker/lease", json={"worker": "w1"}, headers=AUTH
            )
            return response.json() if response.status_code == 200 else None

        deadline = time.monotonic() + 25
        job = None
        while time.monotonic() < deadline and job is None:
            job = leased()
            if job is None:
                time.sleep(0.4)
        assert job is not None, "no analysis job appeared"
        assert job["session"]["venue"] == "Main Gym"
        assert job["calibration"]["mode"] == "near_half"
        assert job["calibration"]["frame_width"] == 1920


class TestLiveFrame:
    def test_frame_available_while_recording(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        response = client.post("/api/sessions", json={"roster": ROSTER})
        session_id = response.json()["session_id"]
        assert (
            client.get(f"/api/sessions/{session_id}/capture/frame").status_code == 404
        )
        video = make_dummy_video(tmp_path / "feed.mp4", seconds=2.5, fps=20)
        client.post(f"/api/sessions/{session_id}/capture", json={"source": str(video)})

        deadline = time.monotonic() + 10
        frame = None
        while time.monotonic() < deadline:
            result = client.get(f"/api/sessions/{session_id}/capture/frame")
            if result.status_code == 200:
                frame = result
                break
            time.sleep(0.3)
        assert frame is not None, "no live frame appeared"
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.content[:2] == b"\xff\xd8"
