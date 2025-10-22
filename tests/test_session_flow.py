from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from session.schemas import ClipCreate, EventIn, RosterEntry, SessionCreate
from session.service import SessionService


def make_service(tmp_path: Path) -> SessionService:
    db_url = f"sqlite:///{tmp_path/'test.db'}"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionService(db_url, sessions_dir)


def test_session_lifecycle(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    session_id = service.create_session(SessionCreate(title="Test", venue="Gym"))
    sessions = service.list_sessions()
    assert sessions[0].title == "Test"

    service.set_roster(
        session_id,
        "A",
        [
            RosterEntry(number="12", player_name="Alice"),
            RosterEntry(number="3", player_name="Bea"),
        ],
    )

    clip_id = service.add_clip(
        session_id, ClipCreate(path="clip.mp4", duration_sec=12.3)
    )
    events = [
        EventIn(t_sec=1.0, event_type="serve", actor_team="A", actor_number="12"),
        EventIn(t_sec=2.5, event_type="attack", actor_team="B", actor_number="8"),
        EventIn(
            t_sec=4.0, event_type="ball_down_in", actor_team="A", actor_number="12"
        ),
    ]
    service.add_events(session_id, clip_id, events)

    rollup = service.get_rollup(session_id)
    assert any(r.event_type == "ball_down_in" for r in rollup)

    service.remove_clip(session_id, clip_id)
    rollup_after = service.get_rollup(session_id)
    assert len(rollup_after) == 0

    export_dir = tmp_path / "exports"
    result = service.export_csv(session_id, str(export_dir))
    assert Path(result["events"]).exists()
    assert Path(result["rollup"]).exists()
