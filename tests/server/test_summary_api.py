"""Summary endpoint: disabled by default, works with an injected client."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hotmama.server.app import create_app
from hotmama.server.config import Settings

from .test_api import append, create_session, start_set


class FakeChat:
    model = "fake-model"

    def complete(self, system: str, user: str) -> str:
        assert "HotMama" in user
        return "Working: serves. Costing: rotation errors. Adjust: swing away."


def test_unconfigured_is_503(tmp_path: Path) -> None:
    with TestClient(
        create_app(Settings(db_path=tmp_path / "t.db", ui_dist=None))
    ) as client:
        session_id = create_session(client)
        response = client.post(f"/api/sessions/{session_id}/summary", json={})
        assert response.status_code == 503


def test_summary_generated_and_persisted(tmp_path: Path) -> None:
    app = create_app(
        Settings(db_path=tmp_path / "t.db", ui_dist=None), chat_client=FakeChat()
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        start_set(client, session_id)
        append(client, session_id, {"type": "rally_ended", "winner": "us", "reason": "kill"})

        response = client.post(
            f"/api/sessions/{session_id}/summary", json={"actor": "coach-phone"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "fake-model"
        assert "Adjust" in body["summary"]

        events = client.get(f"/api/sessions/{session_id}/events").json()
        note = events[-1]
        assert note["type"] == "note_added"
        assert note["text"].startswith("[set summary]")
        assert note["actor"] == "coach-phone"


def test_summary_unknown_session_404(tmp_path: Path) -> None:
    app = create_app(
        Settings(db_path=tmp_path / "t.db", ui_dist=None), chat_client=FakeChat()
    )
    with TestClient(app) as client:
        assert client.post("/api/sessions/s_nope/summary", json={}).status_code == 404
