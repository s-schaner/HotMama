"""Projections over the event-derived state.

Everything here is a pure function of :class:`~hotmama.core.state.MatchState`;
nothing writes, nothing caches. Impact analysis v1 is deliberately
*descriptive* ("4 of your 6 points lost in rotation 3 were serve-receive"),
not correlational — a set is ~25 points and correlation on that sample is
noise dressed up as insight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hotmama.core.events import PointReason, Team
from hotmama.core.state import ROTATIONS, MatchState, PointRecord, RotationTally

_REASON_LABELS = {
    PointReason.KILL.value: "kills",
    PointReason.ACE.value: "aces",
    PointReason.BLOCK.value: "blocks",
    PointReason.ERR_SERVE.value: "serve errors",
    PointReason.ERR_ATTACK.value: "attack errors",
    PointReason.ERR_NET.value: "net faults",
    PointReason.ERR_HANDLING.value: "ball-handling errors",
    PointReason.ERR_OTHER.value: "other errors",
    PointReason.UNKNOWN.value: "unattributed points",
}

# When we LOSE a rally: KILL/ACE/BLOCK were the opponent's winners; ERR_* were ours.
_LOSS_LABELS = {
    PointReason.KILL.value: "opponent kills",
    PointReason.ACE.value: "aces against (serve receive)",
    PointReason.BLOCK.value: "blocked attacks",
    PointReason.ERR_SERVE.value: "our serve errors",
    PointReason.ERR_ATTACK.value: "our attack errors",
    PointReason.ERR_NET.value: "our net faults",
    PointReason.ERR_HANDLING.value: "our ball-handling errors",
    PointReason.ERR_OTHER.value: "our errors (other)",
    PointReason.UNKNOWN.value: "unattributed points",
}


def _all_points(state: MatchState, set_number: int | None = None) -> list[PointRecord]:
    sets = state.sets if set_number is None else [
        s for s in state.sets if s.set_number == set_number
    ]
    return [point for s in sets for point in s.points]


def _merged_rotations(
    state: MatchState, set_number: int | None = None
) -> dict[int, RotationTally]:
    merged = {r: RotationTally() for r in ROTATIONS}
    sets = state.sets if set_number is None else [
        s for s in state.sets if s.set_number == set_number
    ]
    for s in sets:
        for rotation, tally in s.per_rotation.items():
            target = merged[rotation]
            target.serve_won += tally.serve_won
            target.serve_lost += tally.serve_lost
            target.recv_won += tally.recv_won
            target.recv_lost += tally.recv_lost
            for reason, count in tally.lost_reasons.items():
                target.lost_reasons[reason] = target.lost_reasons.get(reason, 0) + count
            for reason, count in tally.won_reasons.items():
                target.won_reasons[reason] = target.won_reasons.get(reason, 0) + count
    return merged


def rotation_table(state: MatchState, set_number: int | None = None) -> list[dict[str, Any]]:
    """Mo's table: one row per rotation with side-out % and the loss breakdown."""
    rows = []
    for rotation, tally in _merged_rotations(state, set_number).items():
        row = tally.to_dict()
        row["rotation"] = rotation
        rows.append(row)
    return rows


def biggest_leak(state: MatchState, set_number: int | None = None) -> dict[str, Any] | None:
    """Impact v1: which (rotation, loss reason) is bleeding the most points.

    Returns the worst leak with a plain-English sentence a coach can act on,
    or None until there is enough signal (>= 3 points lost the same way).
    """
    worst: tuple[int, str, int, int] | None = None  # rotation, reason, count, rot_lost
    for rotation, tally in _merged_rotations(state, set_number).items():
        rotation_lost = tally.serve_lost + tally.recv_lost
        for reason, count in tally.lost_reasons.items():
            if worst is None or count > worst[2]:
                worst = (rotation, reason, count, rotation_lost)
    if worst is None or worst[2] < 3:
        return None
    rotation, reason, count, rotation_lost = worst
    label = _LOSS_LABELS.get(reason, reason)
    sentence = (
        f"Rotation {rotation} is your biggest leak: {count} of the {rotation_lost} "
        f"points lost there came from {label}."
    )
    return {
        "rotation": rotation,
        "reason": reason,
        "count": count,
        "rotation_points_lost": rotation_lost,
        "sentence": sentence,
    }


@dataclass
class _PlayerLine:
    player_id: str
    name: str
    jersey: int | None
    kills: int = 0
    aces: int = 0
    blocks: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    serves: int = 0
    tags: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "jersey": self.jersey,
            "kills": self.kills,
            "aces": self.aces,
            "blocks": self.blocks,
            "errors": dict(self.errors),
            "total_errors": sum(self.errors.values()),
            "serves": self.serves,
            "tags": dict(self.tags),
        }


def player_stats(state: MatchState, set_number: int | None = None) -> list[dict[str, Any]]:
    """Per-player attribution from rally outcomes, serving turns, and tags."""
    lines = {
        pid: _PlayerLine(player_id=pid, name=p.name, jersey=p.jersey)
        for pid, p in state.roster.items()
    }

    for point in _all_points(state, set_number):
        if point.our_server is not None and point.our_server in lines:
            lines[point.our_server].serves += 1
        pid = point.player_id
        if pid is None or pid not in lines:
            continue
        we_won = point.winner is Team.US
        reason = point.reason
        if we_won and reason is PointReason.KILL:
            lines[pid].kills += 1
        elif we_won and reason is PointReason.ACE:
            lines[pid].aces += 1
        elif we_won and reason is PointReason.BLOCK:
            lines[pid].blocks += 1
        elif not we_won and reason.value.startswith("err_"):
            lines[pid].errors[reason.value] = lines[pid].errors.get(reason.value, 0) + 1

    for tag in state.tags:
        if set_number is not None and tag.set_number != set_number:
            continue
        if tag.player_id and tag.player_id in lines:
            key = tag.tag.value
            lines[tag.player_id].tags[key] = lines[tag.player_id].tags.get(key, 0) + 1

    return [line.to_dict() for line in lines.values()]


def scoring_runs(
    state: MatchState, set_number: int | None = None, min_length: int = 3
) -> list[dict[str, Any]]:
    """Consecutive-point runs — where momentum lived."""
    runs: list[dict[str, Any]] = []
    for s in state.sets if set_number is None else [
        x for x in state.sets if x.set_number == set_number
    ]:
        current_team: Team | None = None
        length = 0
        start_score = (0, 0)
        for point in s.points:
            if point.winner is current_team:
                length += 1
            else:
                if current_team is not None and length >= min_length:
                    runs.append(
                        {
                            "set_number": s.set_number,
                            "team": current_team.value,
                            "length": length,
                            "from": f"{start_score[0]}-{start_score[1]}",
                            "to": f"{point.us_points}-{point.them_points}",
                        }
                    )
                current_team = point.winner
                length = 1
                start_score = (point.us_points, point.them_points)
        if current_team is not None and length >= min_length:
            runs.append(
                {
                    "set_number": s.set_number,
                    "team": current_team.value,
                    "length": length,
                    "from": f"{start_score[0]}-{start_score[1]}",
                    "to": f"{s.us_points}-{s.them_points}",
                }
            )
    return runs


def match_summary(state: MatchState) -> dict[str, Any]:
    """Everything the live dashboards and the report share."""
    return {
        "rotation_table": rotation_table(state),
        "biggest_leak": biggest_leak(state),
        "player_stats": player_stats(state),
        "scoring_runs": scoring_runs(state),
        "reason_labels": dict(_REASON_LABELS),
        "loss_labels": dict(_LOSS_LABELS),
    }
