from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_utc TEXT DEFAULT CURRENT_TIMESTAMP,
        title TEXT NOT NULL,
        venue TEXT,
        meta TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        side TEXT NOT NULL CHECK(side IN ('A','B')),
        name TEXT,
        UNIQUE(session_id, side)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roster (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id INTEGER NOT NULL,
        number TEXT NOT NULL,
        player_name TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clips (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        path TEXT NOT NULL,
        duration_sec REAL NOT NULL,
        meta TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        clip_id TEXT NOT NULL,
        t_sec REAL NOT NULL,
        event_type TEXT NOT NULL,
        actor_team TEXT,
        actor_number TEXT,
        actor_resolved_name TEXT,
        payload TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stats_rollup (
        session_id TEXT NOT NULL,
        team_side TEXT,
        number TEXT,
        event_type TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY(session_id, team_side, number, event_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration (
        session_id TEXT NOT NULL,
        clip_id TEXT NOT NULL,
        H TEXT NOT NULL,
        PRIMARY KEY(session_id, clip_id)
    )
    """,
]


class Database:
    def __init__(self, db_url: str) -> None:
        if db_url.startswith("sqlite:///"):
            self.path = Path(db_url.replace("sqlite:///", ""))
        else:
            raise ValueError("Unsupported database URL")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            for statement in SCHEMA:
                conn.execute(statement)
            conn.commit()


__all__ = ["Database"]
