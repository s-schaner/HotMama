"""SQLite event store — the durable append-only log.

One database holds many sessions. The schema carries org/team columns from
day one (tenancy-ready, D7/D11) even though a single gym laptop only ever
writes one team's data; cloud sync later is a data move, not a migration.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from hotmama.core.ids import utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind       TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    org_id     TEXT,
    team_id    TEXT,
    closed     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    seq         INTEGER NOT NULL,
    event_id    TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, seq);
CREATE TABLE IF NOT EXISTS clips (
    clip_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_id   TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'pending',
    path       TEXT,
    url        TEXT,
    start_at   TEXT NOT NULL,
    end_at     TEXT NOT NULL,
    error      TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_session ON clips(session_id, created_at);
"""


class UnknownSessionError(KeyError):
    """The session id does not exist in the store."""


class EventStore:
    """Thread-safe (single lock) SQLite persistence for sessions and events."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        if str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_session(self, session_id: str, *, kind: str, label: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, created_at, kind, label) VALUES (?, ?, ?, ?)",
                (session_id, utcnow().isoformat(), kind, label),
            )
            self._conn.commit()

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.session_id, s.created_at, s.kind, s.label, s.closed,
                       COALESCE(MAX(e.seq), 0) AS last_seq
                FROM sessions s
                LEFT JOIN events e ON e.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_closed(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET closed = 1 WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()

    def append_event(self, session_id: str, event: dict[str, Any]) -> int:
        """Persist one event, returning its store-assigned sequence number."""
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise UnknownSessionError(session_id)
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row["next"])
            self._conn.execute(
                "INSERT INTO events (session_id, seq, event_id, type, payload, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    seq,
                    str(event["event_id"]),
                    str(event["type"]),
                    json.dumps(event, separators=(",", ":")),
                    utcnow().isoformat(),
                ),
            )
            self._conn.commit()
        return seq

    # -- clips ---------------------------------------------------------------

    def insert_clip(
        self,
        clip_id: str,
        *,
        session_id: str,
        event_id: str,
        kind: str,
        label: str,
        start_at: str,
        end_at: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO clips (clip_id, session_id, event_id, kind, label, status,"
                " start_at, end_at, created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                (
                    clip_id,
                    session_id,
                    event_id,
                    kind,
                    label,
                    start_at,
                    end_at,
                    utcnow().isoformat(),
                ),
            )
            self._conn.commit()

    def set_clip_ready(self, clip_id: str, *, path: str, url: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE clips SET status = 'ready', path = ?, url = ?, error = NULL"
                " WHERE clip_id = ?",
                (path, url, clip_id),
            )
            self._conn.commit()

    def set_clip_failed(self, clip_id: str, *, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE clips SET status = 'failed', error = ? WHERE clip_id = ?",
                (error, clip_id),
            )
            self._conn.commit()

    def list_clips(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT clip_id, session_id, event_id, kind, label, status, url,"
                " start_at, end_at, error, created_at FROM clips"
                " WHERE session_id = ? ORDER BY created_at DESC, clip_id DESC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_events(self, session_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        """Events in seq order. Each dict is the stored event payload plus ``seq``."""
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise UnknownSessionError(session_id)
            rows = self._conn.execute(
                "SELECT seq, payload FROM events WHERE session_id = ? AND seq > ?"
                " ORDER BY seq",
                (session_id, after_seq),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["seq"] = int(row["seq"])
            out.append(payload)
        return out
