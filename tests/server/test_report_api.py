"""Report endpoints over a real statted session."""

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
    settings = Settings(
        db_path=tmp_path / "test.db", media_root=tmp_path / "media", ui_dist=None
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _statted_session(client: TestClient) -> str:
    session_id = create_session(client, label="Report Night")
    start_set(client, session_id)
    append(
        client,
        session_id,
        {"type": "rally_ended", "winner": "us", "reason": "kill", "player_id": "p4"},
    )
    append(client, session_id, {"type": "rally_ended", "winner": "them", "reason": "ace"})
    append(client, session_id, {"type": "moment_tagged", "tag": "highlight"})
    return session_id


def test_html_report(client: TestClient) -> None:
    session_id = _statted_session(client)
    response = client.get(f"/api/sessions/{session_id}/report")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Report Night" in response.text
    assert "HotMama" in response.text
    assert "Rotations" in response.text


def test_pdf_report(client: TestClient) -> None:
    pytest.importorskip("weasyprint")
    session_id = _statted_session(client)
    response = client.get(f"/api/sessions/{session_id}/report.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:4] == b"%PDF"
    assert "attachment" in response.headers["content-disposition"]


def test_report_unknown_session(client: TestClient) -> None:
    assert client.get("/api/sessions/s_nope/report").status_code == 404
