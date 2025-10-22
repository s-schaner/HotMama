"""
Session operations module for rollup computation, index building, and data aggregation.

This module provides utilities for:
- Computing statistics rollups from events
- Building search indexes for sessions
- Aggregating session metrics
- Optimizing session queries
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from session import schemas


class SessionIndex:
    """
    In-memory index for fast session lookups and searches.

    Maintains indexes for:
    - Session titles and venues (for search)
    - Event types and counts
    - Player statistics
    - Temporal ordering
    """

    def __init__(self):
        self.sessions: Dict[str, schemas.SessionOut] = {}
        self.title_index: Dict[str, List[str]] = defaultdict(list)
        self.venue_index: Dict[str, List[str]] = defaultdict(list)
        self.date_index: List[Tuple[str, str]] = []  # (created_utc, session_id)

    def add_session(self, session: schemas.SessionOut) -> None:
        """Add a session to the index."""
        self.sessions[session.id] = session

        # Index title keywords
        if session.title:
            for word in session.title.lower().split():
                self.title_index[word].append(session.id)

        # Index venue
        if session.venue:
            venue_key = session.venue.lower()
            self.venue_index[venue_key].append(session.id)

        # Index by date
        self.date_index.append((session.created_utc, session.id))

    def search(self, query: str) -> List[str]:
        """
        Search for sessions by title or venue.

        Args:
            query: Search query string

        Returns:
            List of matching session IDs
        """
        if not query:
            return []

        query_lower = query.lower()
        matches = set()

        # Search titles
        for word in query_lower.split():
            if word in self.title_index:
                matches.update(self.title_index[word])

        # Search venues
        for venue_key, session_ids in self.venue_index.items():
            if query_lower in venue_key:
                matches.update(session_ids)

        # Search by partial session ID match
        for session_id in self.sessions.keys():
            if query_lower in session_id.lower():
                matches.add(session_id)

        return list(matches)

    def get_recent_sessions(self, limit: int = 10) -> List[str]:
        """
        Get most recent session IDs.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session IDs ordered by creation time (newest first)
        """
        sorted_dates = sorted(self.date_index, key=lambda x: x[0], reverse=True)
        return [session_id for _, session_id in sorted_dates[:limit]]


class RollupComputer:
    """
    Computes statistical rollups from raw event data.

    Aggregates events by:
    - Team and player
    - Event type
    - Time windows
    - Success/failure outcomes
    """

    @staticmethod
    def compute_from_events(events: List[schemas.EventIn]) -> Dict[str, Any]:
        """
        Compute comprehensive rollup statistics from events.

        Args:
            events: List of event records

        Returns:
            Dictionary containing aggregated statistics
        """
        rollup = {
            "total_events": len(events),
            "event_types": defaultdict(int),
            "team_stats": defaultdict(lambda: defaultdict(int)),
            "player_stats": defaultdict(lambda: defaultdict(int)),
            "timeline": [],
        }

        for event in events:
            # Count by event type
            rollup["event_types"][event.event_type] += 1

            # Team statistics
            if event.actor_team:
                rollup["team_stats"][event.actor_team][event.event_type] += 1
                rollup["team_stats"][event.actor_team]["total"] += 1

            # Player statistics
            if event.actor_team and event.actor_number:
                player_key = f"{event.actor_team}-{event.actor_number}"
                rollup["player_stats"][player_key][event.event_type] += 1
                rollup["player_stats"][player_key]["total"] += 1
                rollup["player_stats"][player_key]["team"] = event.actor_team
                rollup["player_stats"][player_key]["number"] = event.actor_number
                if event.actor_resolved_name:
                    rollup["player_stats"][player_key][
                        "name"
                    ] = event.actor_resolved_name

        # Convert defaultdicts to regular dicts for JSON serialization
        rollup["event_types"] = dict(rollup["event_types"])
        rollup["team_stats"] = {k: dict(v) for k, v in rollup["team_stats"].items()}
        rollup["player_stats"] = {k: dict(v) for k, v in rollup["player_stats"].items()}

        return rollup

    @staticmethod
    def compute_advanced_metrics(
        events: List[schemas.EventIn],
        rosters: Dict[str, List[schemas.RosterEntry]],
    ) -> Dict[str, Any]:
        """
        Compute advanced performance metrics.

        Args:
            events: List of event records
            rosters: Team rosters for context

        Returns:
            Dictionary containing advanced statistics
        """
        metrics = {
            "attack_efficiency": {},
            "serve_performance": {},
            "defensive_actions": {},
            "rally_stats": {
                "avg_rally_length": 0,
                "longest_rally": 0,
                "total_rallies": 0,
            },
        }

        # Group events by type
        attacks = [e for e in events if e.event_type == "attack"]
        serves = [e for e in events if e.event_type == "serve"]
        defensive = [e for e in events if e.event_type in {"dig", "block", "receive"}]

        # Attack efficiency by player
        attack_by_player = defaultdict(list)
        for attack in attacks:
            if attack.actor_team and attack.actor_number:
                player_key = f"{attack.actor_team}-{attack.actor_number}"
                attack_by_player[player_key].append(attack)

        for player_key, player_attacks in attack_by_player.items():
            kills = sum(1 for a in player_attacks if a.payload.get("outcome") == "kill")
            errors = sum(
                1 for a in player_attacks if a.payload.get("outcome") == "error"
            )
            total = len(player_attacks)
            efficiency = (kills - errors) / total if total > 0 else 0
            metrics["attack_efficiency"][player_key] = {
                "kills": kills,
                "errors": errors,
                "total_attempts": total,
                "efficiency": round(efficiency, 3),
            }

        # Serve performance
        serve_by_player = defaultdict(list)
        for serve in serves:
            if serve.actor_team and serve.actor_number:
                player_key = f"{serve.actor_team}-{serve.actor_number}"
                serve_by_player[player_key].append(serve)

        for player_key, player_serves in serve_by_player.items():
            aces = sum(1 for s in player_serves if s.payload.get("outcome") == "ace")
            errors = sum(
                1 for s in player_serves if s.payload.get("outcome") == "error"
            )
            total = len(player_serves)
            metrics["serve_performance"][player_key] = {
                "aces": aces,
                "errors": errors,
                "total_serves": total,
                "ace_percentage": round(aces / total * 100, 1) if total > 0 else 0,
            }

        # Defensive actions
        for action in defensive:
            if action.actor_team:
                team_key = action.actor_team
                if team_key not in metrics["defensive_actions"]:
                    metrics["defensive_actions"][team_key] = defaultdict(int)
                metrics["defensive_actions"][team_key][action.event_type] += 1

        return metrics

    @staticmethod
    def build_summary(rollup_entries: List[schemas.RollupEntry]) -> Dict[str, Any]:
        """
        Build a human-readable summary from rollup entries.

        Args:
            rollup_entries: List of rollup entries from database

        Returns:
            Summary dictionary with aggregated stats
        """
        summary = {
            "total_events": sum(entry.count for entry in rollup_entries),
            "teams": defaultdict(lambda: {"events": 0, "types": {}}),
            "players": defaultdict(lambda: {"events": 0, "types": {}}),
            "event_types": defaultdict(int),
        }

        for entry in rollup_entries:
            # Event type totals
            summary["event_types"][entry.event_type] += entry.count

            # Team stats
            if entry.team_side:
                summary["teams"][entry.team_side]["events"] += entry.count
                summary["teams"][entry.team_side]["types"][
                    entry.event_type
                ] = entry.count

            # Player stats
            if entry.team_side and entry.number:
                player_key = f"{entry.team_side}-{entry.number}"
                summary["players"][player_key]["events"] += entry.count
                summary["players"][player_key]["types"][entry.event_type] = entry.count
                summary["players"][player_key]["team"] = entry.team_side
                summary["players"][player_key]["number"] = entry.number

        # Convert to regular dicts
        summary["teams"] = {k: dict(v) for k, v in summary["teams"].items()}
        summary["players"] = {k: dict(v) for k, v in summary["players"].items()}
        summary["event_types"] = dict(summary["event_types"])

        return summary


class SessionBackup:
    """
    Handles session backup and recovery operations.

    Provides resilience through:
    - Automatic session snapshots
    - Incremental backups
    - Point-in-time recovery
    """

    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        session_id: str,
        session_data: Dict[str, Any],
    ) -> Path:
        """
        Create a point-in-time snapshot of a session.

        Args:
            session_id: UUID of the session
            session_data: Complete session data to backup

        Returns:
            Path to the snapshot file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.backup_dir / f"{session_id}_{timestamp}.json"

        snapshot = {
            "session_id": session_id,
            "timestamp": timestamp,
            "data": session_data,
        }

        with snapshot_file.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        return snapshot_file

    def list_snapshots(self, session_id: str) -> List[Dict[str, str]]:
        """
        List all snapshots for a session.

        Args:
            session_id: UUID of the session

        Returns:
            List of snapshot metadata
        """
        snapshots = []
        pattern = f"{session_id}_*.json"

        for snapshot_file in self.backup_dir.glob(pattern):
            timestamp_str = snapshot_file.stem.split("_", 1)[1]
            snapshots.append(
                {
                    "path": str(snapshot_file),
                    "timestamp": timestamp_str,
                    "size": snapshot_file.stat().st_size,
                }
            )

        return sorted(snapshots, key=lambda x: x["timestamp"], reverse=True)

    def restore_snapshot(self, snapshot_path: Path) -> Dict[str, Any]:
        """
        Restore a session from a snapshot.

        Args:
            snapshot_path: Path to the snapshot file

        Returns:
            Restored session data
        """
        with snapshot_path.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)

        return snapshot["data"]

    def cleanup_old_snapshots(
        self,
        session_id: str,
        keep_latest: int = 10,
    ) -> int:
        """
        Remove old snapshots, keeping only the most recent ones.

        Args:
            session_id: UUID of the session
            keep_latest: Number of snapshots to keep

        Returns:
            Number of snapshots deleted
        """
        snapshots = self.list_snapshots(session_id)

        if len(snapshots) <= keep_latest:
            return 0

        to_delete = snapshots[keep_latest:]
        deleted = 0

        for snapshot in to_delete:
            try:
                Path(snapshot["path"]).unlink()
                deleted += 1
            except OSError:
                pass

        return deleted
