from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

from app.errors import NotFoundError
from session import schemas
from types import SimpleNamespace

from session.store import Database


class SessionService:
    def __init__(self, db_url: str, sessions_dir: Path) -> None:
        self.db = Database(db_url)
        self.sessions_dir = sessions_dir
        self.engine = SimpleNamespace(url=db_url)

    @contextmanager
    def _connection(self):
        conn = self.db.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_session(self, data: schemas.SessionCreate) -> str:
        session_id = str(uuid.uuid4())
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO sessions(id, title, venue, meta) VALUES (?, ?, ?, ?)",
                (session_id, data.title, data.venue, json.dumps(data.meta)),
            )
            for side in ("A", "B"):
                conn.execute(
                    "INSERT INTO teams(session_id, side, name) VALUES (?, ?, ?)",
                    (session_id, side, None),
                )
        (self.sessions_dir / session_id).mkdir(parents=True, exist_ok=True)
        return session_id

    def list_sessions(self) -> List[schemas.SessionOut]:
        with self._connection() as conn:
            rows = conn.execute("SELECT id, created_utc, title, venue, meta FROM sessions ORDER BY created_utc DESC").fetchall()
        return [
            schemas.SessionOut(
                id=row["id"],
                created_utc=row["created_utc"],
                title=row["title"],
                venue=row["venue"],
                meta=json.loads(row["meta"] or "{}"),
            )
            for row in rows
        ]

    def set_team(self, session_id: str, side: str, name: str | None) -> None:
        with self._connection() as conn:
            updated = conn.execute(
                "UPDATE teams SET name=? WHERE session_id=? AND side=?",
                (name, session_id, side),
            ).rowcount
            if updated == 0:
                raise NotFoundError("Team not found")

    def set_roster(self, session_id: str, side: str, entries: List[schemas.RosterEntry]) -> None:
        with self._connection() as conn:
            team = conn.execute(
                "SELECT id FROM teams WHERE session_id=? AND side=?",
                (session_id, side),
            ).fetchone()
            if not team:
                raise NotFoundError("Team not found")
            team_id = team["id"]
            conn.execute("DELETE FROM roster WHERE team_id=?", (team_id,))
            for entry in entries:
                conn.execute(
                    "INSERT INTO roster(team_id, number, player_name) VALUES (?, ?, ?)",
                    (team_id, entry.number, entry.player_name),
                )

    def add_clip(self, session_id: str, clip: schemas.ClipCreate) -> str:
        clip_id = str(uuid.uuid4())
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO clips(id, session_id, path, duration_sec, meta) VALUES (?, ?, ?, ?, ?)",
                (clip_id, session_id, clip.path, clip.duration_sec, json.dumps(clip.meta)),
            )
        return clip_id

    def add_events(self, session_id: str, clip_id: str, events: List[schemas.EventIn]) -> None:
        with self._connection() as conn:
            for event in events:
                conn.execute(
                    """
                    INSERT INTO events(session_id, clip_id, t_sec, event_type, actor_team, actor_number, actor_resolved_name, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        clip_id,
                        event.t_sec,
                        event.event_type,
                        event.actor_team,
                        event.actor_number,
                        event.actor_resolved_name,
                        json.dumps(event.payload),
                    ),
                )
            self._recompute_rollup(conn, session_id)

    def remove_clip(self, session_id: str, clip_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM events WHERE clip_id=?", (clip_id,))
            conn.execute("DELETE FROM clips WHERE id=?", (clip_id,))
            self._recompute_rollup(conn, session_id)

    def get_rollup(self, session_id: str) -> List[schemas.RollupEntry]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT session_id, team_side, number, event_type, count FROM stats_rollup WHERE session_id=?",
                (session_id,),
            ).fetchall()
        return [
            schemas.RollupEntry(
                session_id=row["session_id"],
                team_side=row["team_side"] or None,
                number=row["number"] or None,
                event_type=row["event_type"],
                count=row["count"],
            )
            for row in rows
        ]

    def rebuild_rollup(self, session_id: str) -> None:
        with self._connection() as conn:
            self._recompute_rollup(conn, session_id)

    def export_csv(self, session_id: str, out_dir: str) -> dict[str, str]:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            events_rows = conn.execute(
                "SELECT clip_id, t_sec, event_type, actor_team, actor_number, actor_resolved_name, payload FROM events WHERE session_id=?",
                (session_id,),
            ).fetchall()
            events_path = output_dir / f"{session_id}_events.csv"
            with events_path.open("w", encoding="utf-8") as f:
                f.write("clip_id,t_sec,event_type,actor_team,actor_number,actor_name,payload\n")
                for row in events_rows:
                    f.write(
                        f"{row['clip_id']},{row['t_sec']},{row['event_type']},{row['actor_team'] or ''},{row['actor_number'] or ''},{row['actor_resolved_name'] or ''},{row['payload']}\n"
                    )
            rollup = self.get_rollup(session_id)
            rollup_path = output_dir / f"{session_id}_rollup.csv"
            with rollup_path.open("w", encoding="utf-8") as f:
                f.write("team_side,number,event_type,count\n")
                for row in rollup:
                    f.write(
                        f"{row.team_side or ''},{row.number or ''},{row.event_type},{row.count}\n"
                    )
        return {"events": str(events_path), "rollup": str(rollup_path)}

    def _recompute_rollup(self, conn, session_id: str) -> None:
        conn.execute("DELETE FROM stats_rollup WHERE session_id=?", (session_id,))
        rows = conn.execute(
            """
            SELECT actor_team, actor_number, event_type, COUNT(*) as cnt
            FROM events
            WHERE session_id=?
            GROUP BY actor_team, actor_number, event_type
            """,
            (session_id,),
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO stats_rollup(session_id, team_side, number, event_type, count) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    row["actor_team"] or "",
                    row["actor_number"] or "",
                    row["event_type"],
                    row["cnt"],
                ),
            )
