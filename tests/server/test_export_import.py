"""Export/import bundles: lossless round-trips with reminted event ids."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hotmama.server.app import create_app
from hotmama.server.config import Settings

from .test_api import append, create_session, start_set


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(
        create_app(Settings(db_path=tmp_path / "t.db", ui_dist=None))
    ) as test_client:
        yield test_client


def _statted_session(client: TestClient) -> str:
    session_id = create_session(client, label="Road game")
    start_set(client, session_id)
    append(client, session_id, {"type": "rally_ended", "winner": "us", "reason": "kill"})
    append(client, session_id, {"type": "rally_ended", "winner": "them", "reason": "ace"})
    append(client, session_id, {"type": "rally_ended", "winner": "us", "reason": "block"})
    append(client, session_id, {"type": "moment_tagged", "tag": "highlight"})
    # An undo, so the bundle contains a retraction whose target must remap.
    assert client.post(f"/api/sessions/{session_id}/undo", json={}).status_code == 200
    return session_id


def test_round_trip_preserves_state(client: TestClient) -> None:
    original_id = _statted_session(client)
    original_state = client.get(f"/api/sessions/{original_id}").json()["state"]

    bundle = client.get(f"/api/sessions/{original_id}/export").json()
    assert bundle["format"] == "hotmama.session.v1"
    assert bundle["session"]["label"] == "Road game"

    imported = client.post("/api/sessions/import", json={"bundle": bundle})
    assert imported.status_code == 201, imported.text
    new_id = imported.json()["session_id"]
    assert new_id != original_id

    new_state = imported.json()["state"]
    original_set = original_state["current_set"]
    new_set = new_state["current_set"]
    assert (new_set["us_points"], new_set["them_points"]) == (
        original_set["us_points"],
        original_set["them_points"],
    )
    assert new_state["warnings"] == []
    # The retraction still bites after id reminting: score is 2-1, not 2-1+tag.
    assert (new_set["us_points"], new_set["them_points"]) == (2, 1)
    assert len(new_state["tags"]) == 0  # undo removed the tag

    # Event ids were reminted, references remapped.
    old_events = client.get(f"/api/sessions/{original_id}/events").json()
    new_events = client.get(f"/api/sessions/{new_id}/events").json()
    assert len(old_events) == len(new_events)
    old_ids = {e["event_id"] for e in old_events}
    new_ids = {e["event_id"] for e in new_events}
    assert old_ids.isdisjoint(new_ids)
    retraction = next(e for e in new_events if e["type"] == "event_retracted")
    assert retraction["target_event_id"] in new_ids


def test_same_bundle_imports_twice(client: TestClient) -> None:
    session_id = _statted_session(client)
    bundle = client.get(f"/api/sessions/{session_id}/export").json()
    first = client.post("/api/sessions/import", json={"bundle": bundle})
    second = client.post("/api/sessions/import", json={"bundle": bundle})
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["session_id"] != second.json()["session_id"]


def test_bad_bundles_rejected(client: TestClient) -> None:
    assert (
        client.post("/api/sessions/import", json={"bundle": {"format": "nope"}})
    ).status_code == 422
    assert (
        client.post(
            "/api/sessions/import",
            json={"bundle": {"format": "hotmama.session.v1", "events": []}},
        )
    ).status_code == 422
    assert (
        client.post(
            "/api/sessions/import",
            json={
                "bundle": {
                    "format": "hotmama.session.v1",
                    "events": [{"type": "made_up"}],
                }
            },
        )
    ).status_code == 422


def test_export_unknown_session_404(client: TestClient) -> None:
    assert client.get("/api/sessions/s_nope/export").status_code == 404
