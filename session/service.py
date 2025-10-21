from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.errors import NotFoundError
from session import schemas
from session.ops import RollupComputer, SessionBackup
from types import SimpleNamespace

from session.store import Database


class SessionService:
    def __init__(self, db_url: str, sessions_dir: Path) -> None:
        self.db = Database(db_url)
        self.sessions_dir = sessions_dir
        self.engine = SimpleNamespace(url=db_url)
        self.backup_manager = SessionBackup(sessions_dir / "_backups")
        self.rollup_computer = RollupComputer()

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

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> List[schemas.SessionOut]:
        """
        List sessions with pagination and optional search.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            search: Optional search term for title/venue

        Returns:
            List of session objects
        """
        with self._connection() as conn:
            if search:
                # Search in title and venue
                query = """
                    SELECT id, created_utc, title, venue, meta
                    FROM sessions
                    WHERE title LIKE ? OR venue LIKE ? OR id LIKE ?
                    ORDER BY created_utc DESC
                    LIMIT ? OFFSET ?
                """
                search_term = f"%{search}%"
                rows = conn.execute(query, (search_term, search_term, search_term, limit, offset)).fetchall()
            else:
                query = """
                    SELECT id, created_utc, title, venue, meta
                    FROM sessions
                    ORDER BY created_utc DESC
                    LIMIT ? OFFSET ?
                """
                rows = conn.execute(query, (limit, offset)).fetchall()

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

    # ========================================================================
    # Enhanced Session Management Methods
    # ========================================================================

    def load_session(self, session_id: str) -> Dict[str, Any]:
        """
        Load complete session data including all related entities.

        Args:
            session_id: UUID of the session

        Returns:
            Dictionary containing session, clips, teams, rosters, and stats

        Raises:
            NotFoundError: If session doesn't exist
        """
        with self._connection() as conn:
            # Load session
            session_row = conn.execute(
                "SELECT id, created_utc, title, venue, meta FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()

            if not session_row:
                raise NotFoundError(f"Session {session_id} not found")

            session = schemas.SessionOut(
                id=session_row["id"],
                created_utc=session_row["created_utc"],
                title=session_row["title"],
                venue=session_row["venue"],
                meta=json.loads(session_row["meta"] or "{}"),
            )

            # Load clips
            clip_rows = conn.execute(
                "SELECT id, session_id, path, duration_sec, meta FROM clips WHERE session_id=?",
                (session_id,),
            ).fetchall()

            clips = [
                schemas.ClipOut(
                    id=row["id"],
                    session_id=row["session_id"],
                    path=row["path"],
                    duration_sec=row["duration_sec"],
                    meta=json.loads(row["meta"] or "{}"),
                )
                for row in clip_rows
            ]

            # Load teams
            team_rows = conn.execute(
                "SELECT side, name FROM teams WHERE session_id=?",
                (session_id,),
            ).fetchall()

            teams = {row["side"]: row["name"] for row in team_rows}

            # Load rosters
            rosters = {}
            for side in ["A", "B"]:
                team = conn.execute(
                    "SELECT id FROM teams WHERE session_id=? AND side=?",
                    (session_id, side),
                ).fetchone()

                if team:
                    roster_rows = conn.execute(
                        "SELECT number, player_name FROM roster WHERE team_id=?",
                        (team["id"],),
                    ).fetchall()

                    rosters[side] = [
                        schemas.RosterEntry(
                            number=row["number"],
                            player_name=row["player_name"],
                        )
                        for row in roster_rows
                    ]
                else:
                    rosters[side] = []

            # Load event count
            event_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE session_id=?",
                (session_id,),
            ).fetchone()["cnt"]

            # Load rollup summary
            rollup = self.get_rollup(session_id)

        rollup_summary = self.rollup_computer.build_summary(rollup)

        return {
            "session": session,
            "clips": clips,
            "teams": teams,
            "rosters": rosters,
            "event_count": event_count,
            "rollup_summary": rollup_summary,
        }

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update session metadata.

        Args:
            session_id: UUID of the session
            updates: Dictionary of fields to update (title, venue, meta)

        Raises:
            NotFoundError: If session doesn't exist
        """
        with self._connection() as conn:
            # Check if session exists
            exists = conn.execute(
                "SELECT id FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()

            if not exists:
                raise NotFoundError(f"Session {session_id} not found")

            # Build update query dynamically
            update_fields = []
            values = []

            if "title" in updates:
                update_fields.append("title=?")
                values.append(updates["title"])

            if "venue" in updates:
                update_fields.append("venue=?")
                values.append(updates["venue"])

            if "meta" in updates:
                update_fields.append("meta=?")
                values.append(json.dumps(updates["meta"]))

            if update_fields:
                query = f"UPDATE sessions SET {', '.join(update_fields)} WHERE id=?"
                values.append(session_id)
                conn.execute(query, tuple(values))

    def delete_session(self, session_id: str, backup: bool = True) -> None:
        """
        Delete a session and all associated data.

        Args:
            session_id: UUID of the session
            backup: Whether to create a backup before deletion

        Raises:
            NotFoundError: If session doesn't exist
        """
        # Create backup if requested
        if backup:
            try:
                session_data = self.load_session(session_id)
                self.backup_manager.create_snapshot(session_id, session_data)
            except Exception:
                # Don't fail deletion if backup fails
                pass

        with self._connection() as conn:
            # Check if session exists
            exists = conn.execute(
                "SELECT id FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()

            if not exists:
                raise NotFoundError(f"Session {session_id} not found")

            # Delete in correct order due to foreign keys
            conn.execute("DELETE FROM events WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM stats_rollup WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM clips WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM calibration WHERE session_id=?", (session_id,))

            # Delete rosters (via team foreign key)
            team_ids = conn.execute(
                "SELECT id FROM teams WHERE session_id=?",
                (session_id,),
            ).fetchall()

            for team in team_ids:
                conn.execute("DELETE FROM roster WHERE team_id=?", (team["id"],))

            conn.execute("DELETE FROM teams WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))

        # Delete session directory
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def attach_artifact(
        self,
        session_id: str,
        artifact_type: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Attach an artifact (heatmap, export, etc.) to a session.

        Args:
            session_id: UUID of the session
            artifact_type: Type of artifact (e.g., 'heatmap', 'export')
            file_path: Path to the artifact file
            metadata: Additional metadata

        Returns:
            Artifact ID (UUID)

        Raises:
            NotFoundError: If session doesn't exist
        """
        with self._connection() as conn:
            # Check if session exists
            exists = conn.execute(
                "SELECT id FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()

            if not exists:
                raise NotFoundError(f"Session {session_id} not found")

        # Create artifacts directory if it doesn't exist
        artifacts_dir = self.sessions_dir / session_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Copy file to artifacts directory
        source_path = Path(file_path)
        artifact_id = str(uuid.uuid4())
        dest_filename = f"{artifact_type}_{artifact_id}{source_path.suffix}"
        dest_path = artifacts_dir / dest_filename

        shutil.copy2(source_path, dest_path)

        # Store metadata
        meta_file = artifacts_dir / f"{artifact_id}.json"
        artifact_meta = {
            "id": artifact_id,
            "type": artifact_type,
            "file": dest_filename,
            "original_path": str(source_path),
            "created_utc": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(artifact_meta, f, indent=2)

        return artifact_id

    def get_session_artifacts(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts attached to a session.

        Args:
            session_id: UUID of the session

        Returns:
            List of artifact metadata dictionaries

        Raises:
            NotFoundError: If session doesn't exist
        """
        with self._connection() as conn:
            # Check if session exists
            exists = conn.execute(
                "SELECT id FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()

            if not exists:
                raise NotFoundError(f"Session {session_id} not found")

        artifacts_dir = self.sessions_dir / session_id / "artifacts"
        if not artifacts_dir.exists():
            return []

        artifacts = []
        for meta_file in artifacts_dir.glob("*.json"):
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    artifact_meta = json.load(f)
                    artifacts.append(artifact_meta)
            except Exception:
                continue

        return sorted(artifacts, key=lambda x: x.get("created_utc", ""), reverse=True)

    def save_session(self, session_id: str, auto_backup: bool = True) -> None:
        """
        Create a snapshot/checkpoint of the current session state.

        Args:
            session_id: UUID of the session
            auto_backup: Whether to create an automatic backup

        Raises:
            NotFoundError: If session doesn't exist
        """
        session_data = self.load_session(session_id)

        if auto_backup:
            self.backup_manager.create_snapshot(session_id, session_data)

            # Cleanup old snapshots (keep latest 10)
            self.backup_manager.cleanup_old_snapshots(session_id, keep_latest=10)
