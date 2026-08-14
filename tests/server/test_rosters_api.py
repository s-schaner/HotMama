"""Saved rosters: type once, reuse all season."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hotmama.server.app import create_app
from hotmama.server.config import Settings

PLAYERS = [
    {"player_id": f"p{i}", "name": f"Player {i}", "jersey": i, "is_libero": i == 12}
    for i in range(1, 13)
]


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(
        create_app(Settings(db_path=tmp_path / "t.db", ui_dist=None))
    ) as test_client:
        yield test_client


def test_save_list_update_delete(client: TestClient) -> None:
    saved = client.put("/api/rosters/HotMama 2026", json={"players": PLAYERS})
    assert saved.status_code == 200
    assert saved.json() == {"name": "HotMama 2026", "players": 12}

    rosters = client.get("/api/rosters").json()
    assert [roster["name"] for roster in rosters] == ["HotMama 2026"]
    assert rosters[0]["players"][11]["is_libero"] is True

    # Upsert: same name replaces, no duplicate rows.
    smaller = PLAYERS[:8]
    client.put("/api/rosters/HotMama 2026", json={"players": smaller})
    rosters = client.get("/api/rosters").json()
    assert len(rosters) == 1
    assert len(rosters[0]["players"]) == 8

    assert client.delete("/api/rosters/HotMama 2026").status_code == 200
    assert client.get("/api/rosters").json() == []
    assert client.delete("/api/rosters/HotMama 2026").status_code == 404


def test_validation(client: TestClient) -> None:
    # Too few players.
    assert (
        client.put("/api/rosters/tiny", json={"players": PLAYERS[:3]})
    ).status_code == 422
    # Duplicate jersey.
    dup_jersey = [*PLAYERS[:6], {**PLAYERS[6], "jersey": 1}]
    assert (
        client.put("/api/rosters/dup", json={"players": dup_jersey})
    ).status_code == 422
    # Duplicate player id.
    dup_id = [*PLAYERS[:6], {**PLAYERS[6], "player_id": "p1", "jersey": 50}]
    assert (
        client.put("/api/rosters/dup2", json={"players": dup_id})
    ).status_code == 422
    # Blank name.
    assert (
        client.put("/api/rosters/%20", json={"players": PLAYERS})
    ).status_code == 422


def test_saved_roster_feeds_session_creation(client: TestClient) -> None:
    client.put("/api/rosters/squad", json={"players": PLAYERS})
    roster = client.get("/api/rosters").json()[0]["players"]
    response = client.post(
        "/api/sessions", json={"our_team": "HotMama", "roster": roster}
    )
    assert response.status_code == 201
    assert len(response.json()["state"]["roster"]) == 12
