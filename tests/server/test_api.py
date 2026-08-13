"""Server contract tests: REST, WebSocket, persistence, error mapping."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hotmama.server.app import create_app
from hotmama.server.config import Settings

ROSTER = [
    {"player_id": f"p{i}", "name": f"Player {i}", "jersey": i, "is_libero": i == 8}
    for i in range(1, 9)
]
LINEUP = ["p1", "p2", "p3", "p4", "p5", "p6"]


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
def client(db_path: Path) -> Iterator[TestClient]:
    app = create_app(Settings(db_path=db_path, ui_dist=None))
    with TestClient(app) as test_client:
        yield test_client


def create_session(client: TestClient, **overrides: Any) -> str:
    body = {
        "kind": "match",
        "our_team": "HotMama",
        "opponent": "Rivals",
        "roster": ROSTER,
        **overrides,
    }
    response = client.post("/api/sessions", json=body)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["state"]["created"] is True
    assert len(data["state"]["roster"]) == len(ROSTER)
    return str(data["session_id"])


def append(client: TestClient, session_id: str, event: dict[str, Any]) -> dict[str, Any]:
    response = client.post(f"/api/sessions/{session_id}/events", json={"event": event})
    assert response.status_code == 200, response.text
    return dict(response.json())


def start_set(client: TestClient, session_id: str, **overrides: Any) -> dict[str, Any]:
    event = {
        "type": "set_started",
        "set_number": 1,
        "lineup": LINEUP,
        "liberos": ["p8"],
        "we_serve_first": True,
        **overrides,
    }
    return append(client, session_id, event)


class TestSessions:
    def test_create_list_get(self, client: TestClient) -> None:
        session_id = create_session(client, label="Tuesday scrimmage")

        listed = client.get("/api/sessions").json()
        assert [s["session_id"] for s in listed] == [session_id]
        assert listed[0]["label"] == "Tuesday scrimmage"
        assert listed[0]["last_seq"] == 2  # session_created + roster_registered

        snapshot = client.get(f"/api/sessions/{session_id}").json()
        assert snapshot["type"] == "snapshot"
        assert snapshot["state"]["our_team"] == "HotMama"
        assert snapshot["summary"]["biggest_leak"] is None

    def test_unknown_session_is_404(self, client: TestClient) -> None:
        assert client.get("/api/sessions/s_nope").status_code == 404
        assert (
            client.post("/api/sessions/s_nope/events", json={"event": {"type": "set_ended"}})
        ).status_code == 404

    def test_bad_best_of_rejected(self, client: TestClient) -> None:
        response = client.post("/api/sessions", json={"best_of": 4})
        assert response.status_code == 422


class TestEventFlow:
    def test_rally_flow_updates_state(self, client: TestClient) -> None:
        session_id = create_session(client)
        start_set(client, session_id)

        payload = append(
            client,
            session_id,
            {"type": "rally_ended", "winner": "us", "reason": "kill", "player_id": "p4"},
        )
        current = payload["state"]["current_set"]
        assert (current["us_points"], current["them_points"]) == (1, 0)
        assert current["serving"] == "us"
        assert payload["seq"] == 4

        payload = append(client, session_id, {"type": "rally_ended", "winner": "them"})
        current = payload["state"]["current_set"]
        assert (current["us_points"], current["them_points"]) == (1, 1)
        assert current["serving"] == "them"

    def test_rule_violation_maps_to_409(self, client: TestClient) -> None:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/events",
            json={"event": {"type": "rally_ended", "winner": "us"}},
        )
        assert response.status_code == 409
        assert "no set in progress" in response.json()["detail"]

    def test_malformed_event_maps_to_422(self, client: TestClient) -> None:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/events",
            json={"event": {"type": "rally_ended", "winner": "aliens"}},
        )
        assert response.status_code == 422

    def test_undo_retracts_last_event(self, client: TestClient) -> None:
        session_id = create_session(client)
        start_set(client, session_id)
        append(client, session_id, {"type": "rally_ended", "winner": "us"})

        response = client.post(f"/api/sessions/{session_id}/undo", json={})
        assert response.status_code == 200
        payload = response.json()
        current = payload["state"]["current_set"]
        assert (current["us_points"], current["them_points"]) == (0, 0)
        assert payload["retracted_event_id"]

        events = client.get(f"/api/sessions/{session_id}/events").json()
        assert events[-1]["type"] == "event_retracted"

    def test_undo_with_nothing_left_is_409(self, client: TestClient) -> None:
        response = client.post("/api/sessions", json={})
        session_id = response.json()["session_id"]
        undo = client.post(f"/api/sessions/{session_id}/undo", json={})
        assert undo.status_code == 409

    def test_actor_is_stamped(self, client: TestClient) -> None:
        session_id = create_session(client)
        start_set(client, session_id)
        client.post(
            f"/api/sessions/{session_id}/events",
            json={"event": {"type": "rally_ended", "winner": "us"}, "actor": "statter-ipad"},
        )
        events = client.get(f"/api/sessions/{session_id}/events").json()
        assert events[-1]["actor"] == "statter-ipad"


class TestPersistence:
    def test_state_survives_restart(self, client: TestClient, db_path: Path) -> None:
        session_id = create_session(client)
        start_set(client, session_id)
        append(client, session_id, {"type": "rally_ended", "winner": "us"})

        fresh_app = create_app(Settings(db_path=db_path, ui_dist=None))
        with TestClient(fresh_app) as fresh_client:
            snapshot = fresh_client.get(f"/api/sessions/{session_id}").json()
            current = snapshot["state"]["current_set"]
            assert (current["us_points"], current["them_points"]) == (1, 0)
            assert snapshot["state"]["warnings"] == []


class TestWebSocket:
    def test_snapshot_broadcast_and_ws_append(self, client: TestClient) -> None:
        session_id = create_session(client)
        start_set(client, session_id)

        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["state"]["current_set"]["set_number"] == 1

            ws.send_json(
                {"type": "append", "event": {"type": "rally_ended", "winner": "us"}}
            )
            update = ws.receive_json()
            assert update["type"] == "event"
            assert update["event"]["type"] == "rally_ended"
            assert update["state"]["current_set"]["us_points"] == 1

            ws.send_json({"type": "append", "event": {"type": "set_ended"}})
            update = ws.receive_json()
            assert update["state"]["sets_won_us"] == 1  # 1-0 when ended: won by us
            assert update["state"]["sets"][0]["finished"] is True

    def test_ws_rejects_bad_event_without_dying(self, client: TestClient) -> None:
        session_id = create_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            ws.receive_json()  # snapshot
            ws.send_json({"type": "append", "event": {"type": "rally_ended", "winner": "us"}})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "no set in progress" in error["detail"]

            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}

    def test_ws_unknown_session_closes(self, client: TestClient) -> None:
        with (
            pytest.raises(Exception),  # noqa: B017 - close surfaces as protocol error
            client.websocket_connect("/ws/sessions/s_nope") as ws,
        ):
            ws.receive_json()
