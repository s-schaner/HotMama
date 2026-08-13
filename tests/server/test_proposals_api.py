"""Confirm-flow over the API: pend → confirm/dismiss, and gated auto-commit."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hotmama.server.app import create_app
from hotmama.server.config import Settings

from .test_api import append, create_session, start_set

TOKEN = "worker-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    values: dict[str, Any] = {
        "db_path": tmp_path / "test.db",
        "media_root": tmp_path / "media",
        "ui_dist": None,
        "worker_token": TOKEN,
        **overrides,
    }
    return TestClient(create_app(Settings(**values)))


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _client(tmp_path) as test_client:
        yield test_client


def _pend_proposal(
    client: TestClient, session_id: str, confidence: float = 0.5
) -> str:
    """Append a proposing observation directly (as a worker would)."""
    payload = append(
        client,
        session_id,
        {
            "type": "cv_observation",
            "kind": "rally_end_detected",
            "producer": "cv_well",
            "confidence": confidence,
            "proposal": {"type": "rally_ended", "winner": "us", "reason": "kill"},
        },
    )
    proposals = payload["state"]["proposals"]
    assert len(proposals) == 1
    return str(proposals[0]["event_id"])


class TestConfirmDismiss:
    def test_confirm_appends_linked_event(self, client: TestClient) -> None:
        session_id = create_session(client)
        start_set(client, session_id)
        observation_id = _pend_proposal(client, session_id)

        response = client.post(
            f"/api/sessions/{session_id}/proposals/{observation_id}/confirm",
            json={"actor": "coach-phone"},
        )
        assert response.status_code == 200
        state = response.json()["state"]
        assert state["proposals"] == []
        assert state["current_set"]["us_points"] == 1

        events = client.get(f"/api/sessions/{session_id}/events").json()
        confirmed = events[-1]
        assert confirmed["type"] == "rally_ended"
        assert confirmed["source_event_id"] == observation_id
        assert confirmed["actor"] == "coach-phone"
        assert confirmed["producer"] == "cv_well"

    def test_dismiss_retracts_observation(self, client: TestClient) -> None:
        session_id = create_session(client)
        start_set(client, session_id)
        observation_id = _pend_proposal(client, session_id)

        response = client.post(
            f"/api/sessions/{session_id}/proposals/{observation_id}/dismiss",
            json={"actor": "coach-phone"},
        )
        assert response.status_code == 200
        state = response.json()["state"]
        assert state["proposals"] == []
        assert state["current_set"]["us_points"] == 0
        assert state["cv_observations"] == 0

    def test_confirm_unknown_proposal_404(self, client: TestClient) -> None:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/proposals/nope/confirm", json={}
        )
        assert response.status_code == 404

    def test_confirm_twice_404s_second_time(self, client: TestClient) -> None:
        session_id = create_session(client)
        start_set(client, session_id)
        observation_id = _pend_proposal(client, session_id)
        first = client.post(
            f"/api/sessions/{session_id}/proposals/{observation_id}/confirm", json={}
        )
        assert first.status_code == 200
        second = client.post(
            f"/api/sessions/{session_id}/proposals/{observation_id}/confirm", json={}
        )
        assert second.status_code == 404

    def test_confirm_rule_breaking_proposal_409_and_stays(
        self, client: TestClient
    ) -> None:
        session_id = create_session(client)  # no set started
        observation_id = _pend_proposal(client, session_id)
        response = client.post(
            f"/api/sessions/{session_id}/proposals/{observation_id}/confirm", json={}
        )
        assert response.status_code == 409
        state = client.get(f"/api/sessions/{session_id}").json()["state"]
        assert len(state["proposals"]) == 1  # still awaiting a valid moment


class TestAutoCommit:
    def test_disabled_by_default(self, client: TestClient) -> None:
        # Default settings: even a 1.0-confidence proposal pends.
        session_id = create_session(client)
        start_set(client, session_id)
        _pend_proposal(client, session_id, confidence=1.0)
        state = client.get(f"/api/sessions/{session_id}").json()["state"]
        assert len(state["proposals"]) == 1
        assert state["current_set"]["us_points"] == 0
