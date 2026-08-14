"""Projection math on an engineered, fully-known match scenario."""

from __future__ import annotations

from hotmama.analytics import (
    biggest_leak,
    match_summary,
    player_stats,
    rotation_table,
    scoring_runs,
)
from hotmama.core import (
    AnyEvent,
    MomentTagged,
    PointReason,
    TagCode,
    Team,
    replay,
)
from tests.core.helpers import base_events, rally


def scenario() -> list[AnyEvent]:
    """Eleven scripted points with known rotations, reasons, and players.

    We serve first (p1 serving, rotation 1):
      1. US   KILL p4         1-0  serve_won[1]
      2. US   ACE  p1         2-0  serve_won[1]
      3. THEM ERR_SERVE p1    2-1  serve_lost[1]  (our serve error)
      4. THEM ACE             2-2  recv_lost[1]   (ace against us)
      5. THEM KILL            2-3  recv_lost[1]
      6. US   KILL p4         3-3  recv_won[1] → side-out → rotation 2
      7. THEM KILL            3-4  serve_lost[2]
      8. US   KILL p4         4-4  recv_won[2] → side-out → rotation 3
      9. THEM ERR_ATTACK p3   4-5  serve_lost[3]
     10. THEM ERR_ATTACK p3   4-6  recv_lost[3]
     11. THEM ERR_ATTACK p5   4-7  recv_lost[3]
    """
    return [
        *base_events(we_serve_first=True),
        rally(Team.US, PointReason.KILL, player_id="p4"),
        rally(Team.US, PointReason.ACE, player_id="p1"),
        rally(Team.THEM, PointReason.ERR_SERVE, player_id="p1"),
        rally(Team.THEM, PointReason.ACE),
        rally(Team.THEM, PointReason.KILL),
        rally(Team.US, PointReason.KILL, player_id="p4"),
        rally(Team.THEM, PointReason.KILL),
        rally(Team.US, PointReason.KILL, player_id="p4"),
        rally(Team.THEM, PointReason.ERR_ATTACK, player_id="p3"),
        rally(Team.THEM, PointReason.ERR_ATTACK, player_id="p3"),
        rally(Team.THEM, PointReason.ERR_ATTACK, player_id="p5"),
        MomentTagged(tag=TagCode.GREAT_DIG, player_id="p6"),
        MomentTagged(tag=TagCode.GREAT_DIG, player_id="p6"),
    ]


def test_rotation_table_splits_serve_and_receive() -> None:
    state = replay(scenario())
    rows = {row["rotation"]: row for row in rotation_table(state)}

    assert rows[1]["serve_won"] == 2
    assert rows[1]["serve_lost"] == 1
    assert rows[1]["recv_won"] == 1
    assert rows[1]["recv_lost"] == 2
    assert rows[1]["side_out_pct"] == 1 / 3
    assert rows[1]["point_diff"] == 0

    assert rows[3]["serve_lost"] == 1
    assert rows[3]["recv_lost"] == 2
    assert rows[3]["lost_reasons"] == {"err_attack": 3}
    assert rows[4]["total"] == 0


def test_biggest_leak_names_rotation_three() -> None:
    state = replay(scenario())
    leak = biggest_leak(state)
    assert leak is not None
    assert leak["rotation"] == 3
    assert leak["reason"] == "err_attack"
    assert leak["count"] == 3
    assert leak["rotation_points_lost"] == 3
    assert "Rotation 3" in leak["sentence"]
    assert "attack errors" in leak["sentence"]


def test_biggest_leak_needs_signal() -> None:
    events = [*base_events(), rally(Team.THEM, PointReason.ERR_ATTACK)]
    assert biggest_leak(replay(events)) is None


def test_player_stats_attribution() -> None:
    state = replay(scenario())
    lines = {line["player_id"]: line for line in player_stats(state)}

    assert lines["p4"]["kills"] == 3
    assert lines["p1"]["aces"] == 1
    assert lines["p1"]["errors"] == {"err_serve": 1}
    assert lines["p3"]["errors"] == {"err_attack": 2}
    assert lines["p5"]["errors"] == {"err_attack": 1}
    # p1 served points 1-3; p2 served point 7; p3 served points 9.
    assert lines["p1"]["serves"] == 3
    assert lines["p2"]["serves"] == 1
    assert lines["p3"]["serves"] == 1
    assert lines["p6"]["tags"] == {"great_dig": 2}


def test_scoring_runs_detected() -> None:
    state = replay(scenario())
    runs = scoring_runs(state)
    assert len(runs) == 2
    assert all(run["team"] == "them" for run in runs)
    assert [run["length"] for run in runs] == [3, 3]


def test_match_summary_bundles_everything() -> None:
    summary = match_summary(replay(scenario()))
    assert {"rotation_table", "biggest_leak", "player_stats", "scoring_runs"} <= set(summary)
