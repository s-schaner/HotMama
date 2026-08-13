"""Shared builders for engine tests."""

from __future__ import annotations

from hotmama.core import (
    AnyEvent,
    PointReason,
    RallyEnded,
    RosterPlayer,
    RosterRegistered,
    SessionCreated,
    SessionKind,
    SetStarted,
    Team,
)

LINEUP = ["p1", "p2", "p3", "p4", "p5", "p6"]
LIBERO = "p8"


def roster_players() -> list[RosterPlayer]:
    return [
        RosterPlayer(player_id=f"p{i}", name=f"Player {i}", jersey=i, is_libero=(i == 8))
        for i in range(1, 9)
    ]


def base_events(
    *, best_of: int = 5, we_serve_first: bool = True, liberos: list[str] | None = None
) -> list[AnyEvent]:
    return [
        SessionCreated(
            kind=SessionKind.MATCH,
            our_team="HotMama",
            opponent="Rivals",
            best_of=best_of,  # type: ignore[arg-type]
        ),
        RosterRegistered(players=roster_players()),
        SetStarted(
            set_number=1,
            lineup=list(LINEUP),
            liberos=[LIBERO] if liberos is None else liberos,
            we_serve_first=we_serve_first,
        ),
    ]


def rally(
    winner: Team,
    reason: PointReason = PointReason.UNKNOWN,
    player_id: str | None = None,
    opponent_jersey: int | None = None,
) -> RallyEnded:
    return RallyEnded(
        winner=winner, reason=reason, player_id=player_id, opponent_jersey=opponent_jersey
    )


def rallies(winner: Team, count: int) -> list[AnyEvent]:
    return [rally(winner) for _ in range(count)]
