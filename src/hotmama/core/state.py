"""Derived state — never stored authoritatively, always replayable from the log."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .events import PointReason, RosterPlayer, SessionKind, TagCode, Team

ROTATIONS = (1, 2, 3, 4, 5, 6)


@dataclass
class RotationTally:
    """Points played while we were in one of our six rotations."""

    serve_won: int = 0
    serve_lost: int = 0
    recv_won: int = 0
    recv_lost: int = 0
    lost_reasons: dict[str, int] = field(default_factory=dict)
    won_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def serve_total(self) -> int:
        return self.serve_won + self.serve_lost

    @property
    def recv_total(self) -> int:
        return self.recv_won + self.recv_lost

    @property
    def total(self) -> int:
        return self.serve_total + self.recv_total

    @property
    def point_diff(self) -> int:
        return (self.serve_won + self.recv_won) - (self.serve_lost + self.recv_lost)

    @property
    def side_out_pct(self) -> float | None:
        """Share of receive rallies we converted (the classic side-out %)."""
        if self.recv_total == 0:
            return None
        return self.recv_won / self.recv_total

    @property
    def serve_win_pct(self) -> float | None:
        if self.serve_total == 0:
            return None
        return self.serve_won / self.serve_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "serve_won": self.serve_won,
            "serve_lost": self.serve_lost,
            "recv_won": self.recv_won,
            "recv_lost": self.recv_lost,
            "total": self.total,
            "point_diff": self.point_diff,
            "side_out_pct": self.side_out_pct,
            "serve_win_pct": self.serve_win_pct,
            "lost_reasons": dict(self.lost_reasons),
            "won_reasons": dict(self.won_reasons),
        }


@dataclass
class PointRecord:
    """One rally's outcome, with full context for later attribution."""

    number: int
    winner: Team
    reason: PointReason
    served_by: Team
    our_rotation: int
    our_server: str | None
    player_id: str | None
    opponent_jersey: int | None
    us_points: int
    them_points: int
    event_id: str
    occurred_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "winner": self.winner.value,
            "reason": self.reason.value,
            "served_by": self.served_by.value,
            "our_rotation": self.our_rotation,
            "our_server": self.our_server,
            "player_id": self.player_id,
            "opponent_jersey": self.opponent_jersey,
            "us_points": self.us_points,
            "them_points": self.them_points,
            "event_id": self.event_id,
        }


@dataclass
class ProposalRecord:
    """A CV observation awaiting a human verdict (confirm or dismiss)."""

    event_id: str
    kind: str
    proposal: dict[str, Any]
    confidence: float
    producer: str
    actor: str | None
    occurred_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "proposal": dict(self.proposal),
            "confidence": self.confidence,
            "producer": self.producer,
            "actor": self.actor,
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass
class TagRecord:
    event_id: str
    tag: TagCode
    player_id: str | None
    note: str | None
    custom_label: str | None
    set_number: int
    us_points: int
    them_points: int
    occurred_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tag": self.tag.value,
            "player_id": self.player_id,
            "note": self.note,
            "custom_label": self.custom_label,
            "set_number": self.set_number,
            "us_points": self.us_points,
            "them_points": self.them_points,
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass
class SetState:
    set_number: int
    to_win: int
    serving: Team
    slot_owner: list[str]
    """Rotation identities by position: index 0 is P1 (the serving slot)."""
    on_court: list[str]
    """Who is physically on the floor per slot (libero swaps change this only)."""
    liberos: list[str] = field(default_factory=list)
    us_points: int = 0
    them_points: int = 0
    rotations_completed: int = 0
    finished: bool = False
    won_by: Team | None = None
    rally_in_progress: bool = False
    rally_started_at: datetime | None = None
    subs_used: int = 0
    timeouts: dict[str, int] = field(default_factory=lambda: {"us": 0, "them": 0})
    per_rotation: dict[int, RotationTally] = field(
        default_factory=lambda: {r: RotationTally() for r in ROTATIONS}
    )
    points: list[PointRecord] = field(default_factory=list)

    @property
    def our_rotation(self) -> int:
        return (self.rotations_completed % 6) + 1

    @property
    def our_server(self) -> str:
        return self.slot_owner[0]

    def rotate(self) -> None:
        """Side-out: everyone shifts one position; old P2 becomes the new server."""
        self.slot_owner = self.slot_owner[1:] + self.slot_owner[:1]
        self.on_court = self.on_court[1:] + self.on_court[:1]
        self.rotations_completed += 1

    @property
    def decided(self) -> Team | None:
        """Set is mathematically over (win-by-2 at target) — awaiting human confirm."""
        hi, lo = max(self.us_points, self.them_points), min(self.us_points, self.them_points)
        if hi >= self.to_win and hi - lo >= 2:
            return Team.US if self.us_points > self.them_points else Team.THEM
        return None

    @property
    def set_point(self) -> Team | None:
        """A team wins the set by taking the next rally."""
        if self.decided is not None or self.finished:
            return None
        for team, mine, theirs in (
            (Team.US, self.us_points, self.them_points),
            (Team.THEM, self.them_points, self.us_points),
        ):
            if mine + 1 >= self.to_win and mine + 1 - theirs >= 2:
                return team
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_number": self.set_number,
            "to_win": self.to_win,
            "us_points": self.us_points,
            "them_points": self.them_points,
            "serving": self.serving.value,
            "our_rotation": self.our_rotation,
            "our_server": self.our_server,
            "slot_owner": list(self.slot_owner),
            "on_court": list(self.on_court),
            "liberos": list(self.liberos),
            "finished": self.finished,
            "won_by": self.won_by.value if self.won_by else None,
            "decided": self.decided.value if self.decided else None,
            "set_point": self.set_point.value if self.set_point else None,
            "rally_in_progress": self.rally_in_progress,
            "subs_used": self.subs_used,
            "timeouts": dict(self.timeouts),
            "per_rotation": {r: t.to_dict() for r, t in self.per_rotation.items()},
            "points": [p.to_dict() for p in self.points[-10:]],
            "total_points": len(self.points),
        }


@dataclass
class MatchState:
    created: bool = False
    kind: SessionKind = SessionKind.MATCH
    our_team: str = "Us"
    opponent: str = "Them"
    best_of: int = 5
    set_points: int = 25
    final_set_points: int = 15
    roster: dict[str, RosterPlayer] = field(default_factory=dict)
    sets: list[SetState] = field(default_factory=list)
    tags: list[TagRecord] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cv_observations: int = 0
    proposals: dict[str, ProposalRecord] = field(default_factory=dict)
    """Pending CV proposals by observation event_id — confirm or dismiss."""
    session_closed: bool = False
    applied_events: int = 0
    last_event_id: str | None = None

    @property
    def current_set(self) -> SetState | None:
        if self.sets and not self.sets[-1].finished:
            return self.sets[-1]
        return None

    @property
    def sets_won_us(self) -> int:
        return sum(1 for s in self.sets if s.won_by is Team.US)

    @property
    def sets_won_them(self) -> int:
        return sum(1 for s in self.sets if s.won_by is Team.THEM)

    @property
    def sets_to_win_match(self) -> int:
        return self.best_of // 2 + 1

    @property
    def match_over(self) -> bool:
        need = self.sets_to_win_match
        return self.sets_won_us >= need or self.sets_won_them >= need

    def to_public_dict(self) -> dict[str, Any]:
        """The wire snapshot the UI renders. The UI never re-derives rules."""
        current = self.current_set
        return {
            "created": self.created,
            "kind": self.kind.value,
            "our_team": self.our_team,
            "opponent": self.opponent,
            "best_of": self.best_of,
            "set_points": self.set_points,
            "final_set_points": self.final_set_points,
            "roster": [p.model_dump() for p in self.roster.values()],
            "sets_won_us": self.sets_won_us,
            "sets_won_them": self.sets_won_them,
            "match_over": self.match_over,
            "session_closed": self.session_closed,
            "current_set": current.to_dict() if current else None,
            "sets": [s.to_dict() for s in self.sets],
            "tags": [t.to_dict() for t in self.tags],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "cv_observations": self.cv_observations,
            "proposals": [
                record.to_dict()
                for record in sorted(self.proposals.values(), key=lambda r: r.occurred_at)
            ],
            "applied_events": self.applied_events,
            "last_event_id": self.last_event_id,
        }
