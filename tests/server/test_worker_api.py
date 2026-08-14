"""Pull-worker feed: auth, lease semantics, full loop with the reference worker."""

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
from hotmama.worker import StubEngine, WorkerClient  # noqa: E402

from ..capture.util import make_dummy_video  # noqa: E402
from .test_api import LINEUP, ROSTER  # noqa: E402

TOKEN = "test-worker-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "db_path": tmp_path / "test.db",
        "media_root": tmp_path / "media",
        "ui_dist": None,
        "segment_seconds": 1.0,
        "rally_pad_seconds": 0.3,
        "rally_max_seconds": 5.0,
        "worker_token": TOKEN,
        "worker_lease_seconds": 300.0,
        **overrides,
    }
    return Settings(**values)


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(tmp_path))) as test_client:
        yield test_client


def _poll(check: Any, timeout: float, message: str) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result is not None:
            return result
        time.sleep(0.3)
    pytest.fail(f"timed out: {message}")


def _session_with_rally_chunk(client: TestClient, tmp_path: Path) -> str:
    """Record a short session with one rally so one analysis job exists."""
    response = client.post(
        "/api/sessions",
        json={"our_team": "HotMama", "opponent": "Rivals", "roster": ROSTER},
    )
    session_id = str(response.json()["session_id"])
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
    video = make_dummy_video(tmp_path / "feed.mp4", seconds=2.5, fps=20)
    assert (
        client.post(
            f"/api/sessions/{session_id}/capture",
            json={"source": str(video), "tag_clips": False},
        ).status_code
        == 200
    )
    time.sleep(0.8)
    client.post(
        f"/api/sessions/{session_id}/events",
        json={"event": {"type": "rally_ended", "winner": "us", "reason": "kill"}},
    )

    def rally_clip_ready() -> bool | None:
        clips = client.get(f"/api/sessions/{session_id}/clips").json()
        ready = [c for c in clips if c["kind"] == "rally" and c["status"] == "ready"]
        return True if ready else None

    _poll(rally_clip_ready, timeout=25.0, message="rally chunk never became ready")
    return session_id


class TestAuth:
    def test_disabled_feed_is_503(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, worker_token=None)
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/api/worker/lease", json={"worker": "w1"}, headers=AUTH
            )
            assert response.status_code == 503

    def test_bad_token_is_401(self, client: TestClient) -> None:
        response = client.post(
            "/api/worker/lease",
            json={"worker": "w1"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401

    def test_missing_token_is_401(self, client: TestClient) -> None:
        assert (
            client.post("/api/worker/lease", json={"worker": "w1"}).status_code == 401
        )


class TestLease:
    def test_empty_queue_is_204(self, client: TestClient) -> None:
        response = client.post("/api/worker/lease", json={"worker": "w1"}, headers=AUTH)
        assert response.status_code == 204

    def test_lease_complete_loop(self, client: TestClient, tmp_path: Path) -> None:
        session_id = _session_with_rally_chunk(client, tmp_path)

        lease = client.post("/api/worker/lease", json={"worker": "w1"}, headers=AUTH)
        assert lease.status_code == 200
        job = lease.json()
        assert job["session_id"] == session_id
        assert job["kind"] == "rally"
        assert job["session"]["label"] == "HotMama vs Rivals"

        # The chunk downloads through the same host.
        media = client.get(job["clip_url"])
        assert media.status_code == 200 and len(media.content) > 0

        # While leased, nothing else is available.
        assert (
            client.post("/api/worker/lease", json={"worker": "w2"}, headers=AUTH)
        ).status_code == 204

        done = client.post(
            "/api/worker/complete",
            json={
                "clip_id": job["clip_id"],
                "session_id": session_id,
                "worker": "w1",
                "observations": [
                    {"kind": "clip_stats", "data": {"frames": 12}, "confidence": 0.9},
                    {"kind": "ball_seen", "data": {"zone": 5}, "confidence": 0.4},
                ],
            },
            headers=AUTH,
        )
        assert done.status_code == 200
        assert done.json() == {"status": "done", "appended": 2, "auto_committed": 0}

        events = client.get(f"/api/sessions/{session_id}/events").json()
        observations = [e for e in events if e["type"] == "cv_observation"]
        assert len(observations) == 2
        assert observations[0]["producer"] == "cv_well"
        assert observations[0]["actor"] == "w1"
        assert observations[0]["data"]["clip_id"] == job["clip_id"]

        state = client.get(f"/api/sessions/{session_id}").json()["state"]
        assert state["cv_observations"] == 2
        assert client.get(f"/api/sessions/{session_id}/analysis").json() == {"done": 1}
        assert (
            client.post("/api/worker/lease", json={"worker": "w1"}, headers=AUTH)
        ).status_code == 204

    def test_failure_requeues_then_fails(self, client: TestClient, tmp_path: Path) -> None:
        session_id = _session_with_rally_chunk(client, tmp_path)
        clip_id = None
        for attempt in range(3):
            lease = client.post(
                "/api/worker/lease", json={"worker": "w1"}, headers=AUTH
            )
            assert lease.status_code == 200, f"attempt {attempt} got no job"
            clip_id = lease.json()["clip_id"]
            response = client.post(
                "/api/worker/complete",
                json={
                    "clip_id": clip_id,
                    "session_id": session_id,
                    "worker": "w1",
                    "ok": False,
                    "error": "model exploded",
                },
                headers=AUTH,
            )
            assert response.status_code == 200

        assert (
            client.post("/api/worker/lease", json={"worker": "w1"}, headers=AUTH)
        ).status_code == 204
        assert client.get(f"/api/sessions/{session_id}/analysis").json() == {
            "failed": 1
        }

    def test_cloud_producer_allowed_human_rejected(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        session_id = _session_with_rally_chunk(client, tmp_path)
        job = client.post(
            "/api/worker/lease", json={"worker": "w1"}, headers=AUTH
        ).json()
        rejected = client.post(
            "/api/worker/complete",
            json={
                "clip_id": job["clip_id"],
                "session_id": session_id,
                "worker": "w1",
                "producer": "human",
                "observations": [],
            },
            headers=AUTH,
        )
        assert rejected.status_code == 422


class TestAutoCommit:
    def test_high_confidence_commits_low_confidence_pends(
        self, tmp_path: Path
    ) -> None:
        with TestClient(
            create_app(_settings(tmp_path, auto_commit_confidence=0.8))
        ) as client:
            session_id = _session_with_rally_chunk(client, tmp_path)
            job = client.post(
                "/api/worker/lease", json={"worker": "w1"}, headers=AUTH
            ).json()

            done = client.post(
                "/api/worker/complete",
                json={
                    "clip_id": job["clip_id"],
                    "session_id": session_id,
                    "worker": "w1",
                    "observations": [
                        {
                            "kind": "rally_end_detected",
                            "confidence": 0.95,
                            "proposal": {
                                "type": "rally_ended",
                                "winner": "us",
                                "reason": "kill",
                            },
                        },
                        {
                            "kind": "rally_end_detected",
                            "confidence": 0.4,
                            "proposal": {
                                "type": "rally_ended",
                                "winner": "them",
                                "reason": "ace",
                            },
                        },
                    ],
                },
                headers=AUTH,
            )
            assert done.status_code == 200
            body = done.json()
            assert body["appended"] == 2
            assert body["auto_committed"] == 1

            state = client.get(f"/api/sessions/{session_id}").json()["state"]
            # One rally was already statted manually in the helper (1-0);
            # the 0.95 proposal auto-committed another point for us.
            assert state["current_set"]["us_points"] == 2
            assert len(state["proposals"]) == 1
            assert state["proposals"][0]["confidence"] == 0.4

            events = client.get(f"/api/sessions/{session_id}/events").json()
            committed = [
                e
                for e in events
                if e["type"] == "rally_ended" and e.get("source_event_id")
            ]
            assert len(committed) == 1
            assert committed[0]["producer"] == "cv_well"


class TestVlmWorkerLoop:
    def test_vlm_engine_full_loop(self, client: TestClient, tmp_path: Path) -> None:
        import httpx

        from hotmama.worker.vlm import VlmClient, VlmEngine

        session_id = _session_with_rally_chunk(client, tmp_path)

        summary = {
            "rally_visible": True,
            "description": "Rally ends with a ball down near side.",
            "ball_landed": "near",
            "serve_visible": True,
            "jersey_numbers": [9],
            "confidence": 0.66,
        }

        def vision_handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            body = _json.loads(request.content)
            assert body["model"] == "qwen3-vl-8b"
            assert any(
                part["type"] == "image_url"
                for part in body["messages"][0]["content"][1:]
            )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": _json.dumps(summary)}}]},
            )

        vlm_client = VlmClient(
            base_url="http://corona:8005",
            model="qwen3-vl-8b",
            http=httpx.Client(transport=httpx.MockTransport(vision_handler)),
        )
        worker = WorkerClient(
            token=TOKEN,
            worker_name="corona-worker",
            engine=VlmEngine(vlm_client, frame_count=4),
            http=client,
        )
        assert worker.run_once() is True

        events = client.get(f"/api/sessions/{session_id}/events").json()
        observation = next(e for e in events if e["type"] == "cv_observation")
        assert observation["kind"] == "vlm_rally_summary"
        assert observation["confidence"] == 0.66
        assert observation["data"]["jersey_numbers"] == [9]
        assert observation["actor"] == "corona-worker"
        assert client.get(f"/api/sessions/{session_id}/analysis").json() == {"done": 1}


class TestReferenceWorker:
    def test_worker_client_full_loop(self, client: TestClient, tmp_path: Path) -> None:
        session_id = _session_with_rally_chunk(client, tmp_path)
        worker = WorkerClient(
            token=TOKEN, worker_name="test-box", engine=StubEngine(), http=client
        )
        assert worker.run_once() is True
        assert worker.run_once() is False  # queue drained

        state = client.get(f"/api/sessions/{session_id}").json()["state"]
        assert state["cv_observations"] == 1
        events = client.get(f"/api/sessions/{session_id}/events").json()
        observation = next(e for e in events if e["type"] == "cv_observation")
        assert observation["kind"] == "clip_stats"
        assert observation["data"]["frames"] > 0
        assert observation["actor"] == "test-box"
        assert client.get(f"/api/sessions/{session_id}/analysis").json() == {"done": 1}
