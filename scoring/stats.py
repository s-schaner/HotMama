from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# Volleyball event type definitions
class EventType:
    """Volleyball event types for comprehensive analytics."""

    # Offensive actions
    SERVE = "serve"
    SERVE_ACE = "serve_ace"
    SERVE_ERROR = "serve_error"
    ATTACK = "attack"
    KILL = "kill"
    ATTACK_ERROR = "attack_error"
    ATTACK_BLOCKED = "attack_blocked"
    SET = "set"
    ASSIST = "assist"

    # Defensive actions
    DIG = "dig"
    DIG_ERROR = "dig_error"
    BLOCK = "block"
    BLOCK_SOLO = "block_solo"
    BLOCK_ASSIST = "block_assist"
    BLOCK_ERROR = "block_error"
    RECEIVE = "receive"
    RECEIVE_ERROR = "receive_error"

    # Ball handling
    TOUCH = "touch"
    PASS = "pass"
    BUMP = "bump"
    OVERPASS = "overpass"

    # Errors
    ROTATION_ERROR = "rotation_error"
    FOOT_FAULT = "foot_fault"
    NET_VIOLATION = "net_violation"
    DOUBLE_CONTACT = "double_contact"
    LIFT = "lift"
    CARRY = "carry"

    # Court events
    BALL_IN = "ball_in"
    BALL_OUT = "ball_out"
    POINT_SCORED = "point_scored"
    POINT_LOST = "point_lost"

    # Movement/Position
    JUMP = "jump"
    SPIKE_APPROACH = "spike_approach"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"


class StatsAccumulator:
    """
    Comprehensive volleyball statistics accumulator.

    Tracks:
    - Event counts per player
    - Success/error rates
    - Advanced metrics (efficiency, rating, etc.)
    - Temporal patterns
    """

    def __init__(self) -> None:
        """Initialize stats accumulator."""
        # Basic event counter: (team, number, event_type) -> count
        self.counter = Counter()

        # Per-player detailed stats: track_id -> {metric: value}
        self.player_stats = defaultdict(lambda: defaultdict(int))

        # Track to player mapping: track_id -> (team, number)
        self.track_to_player = {}

        # Temporal stats: track_id -> [(timestamp, event_type), ...]
        self.temporal_events = defaultdict(list)

        logger.info("StatsAccumulator initialized")

    def ingest(self, events: Iterable[dict]) -> None:
        """
        Ingest events into stats accumulator.

        Expected event dict keys:
        - actor_team: str
        - actor_number: str or int
        - event_type: str
        - track_id: int (optional)
        - t_sec: float (optional, timestamp)
        """
        for event in events:
            team = event.get("actor_team", "")
            number = str(event.get("actor_number", ""))
            event_type = event.get("event_type", "unknown")
            track_id = event.get("track_id")
            timestamp = event.get("t_sec")

            # Basic counter
            key = (team, number, event_type)
            self.counter[key] += 1

            # Track to player mapping
            if track_id is not None:
                self.track_to_player[track_id] = (team, number)

                # Per-player stats
                self.player_stats[track_id][event_type] += 1

                # Temporal tracking
                if timestamp is not None:
                    self.temporal_events[track_id].append((timestamp, event_type))

    def snapshot(self) -> List[dict]:
        """
        Get snapshot of current stats.

        Returns:
            List of stat dicts with keys: actor_team, actor_number, event_type, count
        """
        return [
            {
                "actor_team": team,
                "actor_number": number,
                "event_type": event_type,
                "count": count,
            }
            for (team, number, event_type), count in self.counter.items()
        ]

    def get_player_summary(self, team: str, number: str) -> Dict[str, int]:
        """Get summary stats for a specific player."""
        summary = {}

        for (t, n, event_type), count in self.counter.items():
            if t == team and n == number:
                summary[event_type] = count

        return summary

    def get_track_summary(self, track_id: int) -> Dict[str, int]:
        """Get summary stats for a specific track ID."""
        return dict(self.player_stats[track_id])

    def compute_advanced_metrics(self, track_id: int) -> Dict[str, float]:
        """
        Compute advanced volleyball metrics for a player.

        Metrics:
        - Kill percentage
        - Attack efficiency
        - Serve efficiency
        - Dig percentage
        - Block efficiency
        - Overall rating
        """
        stats = self.player_stats[track_id]

        # Attack metrics
        kills = stats.get(EventType.KILL, 0)
        attacks = stats.get(EventType.ATTACK, 0) + kills
        attack_errors = stats.get(EventType.ATTACK_ERROR, 0)
        attack_blocked = stats.get(EventType.ATTACK_BLOCKED, 0)

        kill_percentage = (kills / attacks * 100) if attacks > 0 else 0.0
        attack_efficiency = (
            ((kills - attack_errors - attack_blocked) / attacks) if attacks > 0 else 0.0
        )

        # Serve metrics
        serve_aces = stats.get(EventType.SERVE_ACE, 0)
        serves = stats.get(EventType.SERVE, 0) + serve_aces
        serve_errors = stats.get(EventType.SERVE_ERROR, 0)

        serve_efficiency = ((serve_aces - serve_errors) / serves) if serves > 0 else 0.0

        # Defensive metrics
        digs = stats.get(EventType.DIG, 0)
        dig_errors = stats.get(EventType.DIG_ERROR, 0)
        dig_attempts = digs + dig_errors

        dig_percentage = (digs / dig_attempts * 100) if dig_attempts > 0 else 0.0

        # Block metrics
        blocks = (
            stats.get(EventType.BLOCK, 0)
            + stats.get(EventType.BLOCK_SOLO, 0)
            + stats.get(EventType.BLOCK_ASSIST, 0)
        )
        block_errors = stats.get(EventType.BLOCK_ERROR, 0)

        # Overall rating (simplified)
        rating = (
            kills * 3.0
            + serve_aces * 2.0
            + digs * 1.5
            + blocks * 2.5
            + stats.get(EventType.ASSIST, 0) * 2.0
            - attack_errors * 2.0
            - serve_errors * 1.5
            - dig_errors * 1.0
            - block_errors * 1.0
        )

        return {
            "kill_percentage": round(kill_percentage, 2),
            "attack_efficiency": round(attack_efficiency, 3),
            "serve_efficiency": round(serve_efficiency, 3),
            "dig_percentage": round(dig_percentage, 2),
            "total_blocks": blocks,
            "rating": round(rating, 2),
        }

    def get_all_player_summaries(self) -> Dict[Tuple[str, str], Dict]:
        """
        Get summaries for all players.

        Returns:
            Dict mapping (team, number) to stats dict
        """
        summaries = {}

        for (team, number), track_id in {
            (t, n): tid for tid, (t, n) in self.track_to_player.items()
        }.items():
            basic_stats = self.get_player_summary(team, number)
            advanced_metrics = self.compute_advanced_metrics(track_id)

            summaries[(team, number)] = {
                "basic_stats": basic_stats,
                "advanced_metrics": advanced_metrics,
            }

        return summaries

    def get_team_summary(self, team: str) -> Dict[str, int]:
        """Get aggregated stats for a team."""
        team_stats = defaultdict(int)

        for (t, n, event_type), count in self.counter.items():
            if t == team:
                team_stats[event_type] += count

        return dict(team_stats)

    def get_temporal_pattern(self, track_id: int) -> List[Tuple[float, str]]:
        """Get temporal event pattern for a player."""
        return sorted(self.temporal_events[track_id])

    def export_to_csv_format(self) -> List[Dict]:
        """
        Export stats in CSV-friendly format.

        Returns:
            List of dicts suitable for CSV export
        """
        rows = []

        for track_id, (team, number) in self.track_to_player.items():
            stats = self.get_track_summary(track_id)
            advanced = self.compute_advanced_metrics(track_id)

            row = {
                "track_id": track_id,
                "team": team,
                "number": number,
                **stats,
                **advanced,
            }

            rows.append(row)

        return rows

    def reset(self):
        """Reset all stats."""
        self.counter.clear()
        self.player_stats.clear()
        self.track_to_player.clear()
        self.temporal_events.clear()
        logger.info("Stats reset")
