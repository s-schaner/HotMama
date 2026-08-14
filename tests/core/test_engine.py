"""Engine rules: scoring, rotation, sets, corrections. The trust layer."""

from __future__ import annotations

import pytest

from hotmama.core import (
    EngineError,
    EventRetracted,
    LiberoSwap,
    MomentTagged,
    PointReason,
    RallyStarted,
    ScoreAdjusted,
    SessionClosed,
    SessionCreated,
    SetEnded,
    SetStarted,
    SubMade,
    TagCode,
    Team,
    TimeoutCalled,
    apply_strict,
    replay,
    retractable_event,
)

from .helpers import LINEUP, base_events, rallies, rally


class TestSessionLifecycle:
    def test_events_before_session_are_rejected(self) -> None:
        with pytest.raises(EngineError, match="session not created"):
            apply_strict([], MomentTagged(tag=TagCode.HIGHLIGHT))

    def test_second_session_created_rejected(self) -> None:
        events = base_events()
        with pytest.raises(EngineError, match="already created"):
            apply_strict(events, SessionCreated())

    def test_closed_session_blocks_new_set(self) -> None:
        events = [*base_events(), *rallies(Team.US, 25), SetEnded(), SessionClosed()]
        with pytest.raises(EngineError, match="closed"):
            apply_strict(events, SetStarted(set_number=2, lineup=list(LINEUP)))


class TestScoringAndServing:
    def test_win_on_our_serve_holds_rotation(self) -> None:
        state = replay([*base_events(we_serve_first=True), rally(Team.US)])
        current = state.current_set
        assert current is not None
        assert (current.us_points, current.them_points) == (1, 0)
        assert current.serving is Team.US
        assert current.our_rotation == 1
        assert current.our_server == "p1"
        assert current.per_rotation[1].serve_won == 1

    def test_side_out_rotates_and_takes_serve(self) -> None:
        state = replay([*base_events(we_serve_first=False), rally(Team.US)])
        current = state.current_set
        assert current is not None
        assert current.per_rotation[1].recv_won == 1
        assert current.our_rotation == 2
        assert current.our_server == "p2"
        assert current.serving is Team.US

    def test_losing_our_serve_does_not_rotate_us(self) -> None:
        state = replay([*base_events(we_serve_first=True), rally(Team.THEM)])
        current = state.current_set
        assert current is not None
        assert (current.us_points, current.them_points) == (0, 1)
        assert current.serving is Team.THEM
        assert current.our_rotation == 1
        assert current.per_rotation[1].serve_lost == 1

    def test_six_side_outs_complete_a_full_cycle(self) -> None:
        events = base_events(we_serve_first=False)
        for _ in range(6):
            events.append(rally(Team.US))    # side-out: rotate, we serve
            events.append(rally(Team.THEM))  # they take the serve back
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.our_rotation == 1
        assert current.slot_owner == LINEUP
        assert (current.us_points, current.them_points) == (6, 6)

    def test_point_record_captures_rally_context(self) -> None:
        state = replay(
            [
                *base_events(we_serve_first=True),
                rally(Team.US, PointReason.KILL, player_id="p4", opponent_jersey=12),
            ]
        )
        current = state.current_set
        assert current is not None
        point = current.points[0]
        assert point.winner is Team.US
        assert point.reason is PointReason.KILL
        assert point.served_by is Team.US
        assert point.our_server == "p1"
        assert point.player_id == "p4"
        assert point.opponent_jersey == 12
        assert (point.us_points, point.them_points) == (1, 0)

    def test_rally_player_must_be_on_roster(self) -> None:
        with pytest.raises(EngineError, match="not in roster"):
            apply_strict(base_events(), rally(Team.US, PointReason.KILL, player_id="ghost"))


class TestSetAndMatchRules:
    def test_win_by_two_flags(self) -> None:
        events = [*base_events(), *rallies(Team.US, 24), *rallies(Team.THEM, 23)]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.decided is None
        assert current.set_point is Team.US

        state = replay([*events, rally(Team.US)])
        current = state.current_set
        assert current is not None
        assert current.decided is Team.US

    def test_deuce_requires_two_point_margin(self) -> None:
        events = [*base_events(), *rallies(Team.US, 24), *rallies(Team.THEM, 24)]
        state = replay([*events, rally(Team.US)])
        current = state.current_set
        assert current is not None
        assert (current.us_points, current.them_points) == (25, 24)
        assert current.decided is None
        assert current.set_point is Team.US

        state = replay([*events, rally(Team.US), rally(Team.US)])
        current = state.current_set
        assert current is not None
        assert current.decided is Team.US

    def test_rally_after_decided_set_is_rejected(self) -> None:
        events = [*base_events(), *rallies(Team.US, 25)]
        with pytest.raises(EngineError, match="decided"):
            apply_strict(events, rally(Team.US))

    def test_set_ended_finalizes_and_next_set_starts(self) -> None:
        events = [*base_events(), *rallies(Team.US, 25), SetEnded()]
        state = replay(events)
        assert state.sets_won_us == 1
        assert state.current_set is None

        events.append(SetStarted(set_number=2, lineup=list(LINEUP), we_serve_first=False))
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.set_number == 2
        assert current.serving is Team.THEM

    def test_set_ended_on_tie_rejected(self) -> None:
        events = [*base_events(), rally(Team.US), rally(Team.THEM)]
        with pytest.raises(EngineError, match="tied"):
            apply_strict(events, SetEnded())

    def test_wrong_set_number_rejected(self) -> None:
        events = [*base_events(), *rallies(Team.US, 25), SetEnded()]
        with pytest.raises(EngineError, match="expected set 2"):
            apply_strict(events, SetStarted(set_number=3, lineup=list(LINEUP)))

    def test_fifth_set_plays_to_final_set_points(self) -> None:
        events = base_events(best_of=5)
        for set_number in range(1, 5):
            winner = Team.US if set_number % 2 else Team.THEM
            events.extend(rallies(winner, 25))
            events.append(SetEnded())
            events.append(SetStarted(set_number=set_number + 1, lineup=list(LINEUP)))
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.set_number == 5
        assert current.to_win == 15

        events.extend(rallies(Team.US, 15))
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.decided is Team.US

    def test_match_over_blocks_further_sets(self) -> None:
        events = base_events(best_of=3)
        for set_number in (1, 2):
            events.extend(rallies(Team.US, 25))
            events.append(SetEnded())
            if set_number == 1:
                events.append(SetStarted(set_number=2, lineup=list(LINEUP)))
        state = replay(events)
        assert state.match_over
        with pytest.raises(EngineError, match="decided"):
            apply_strict(events, SetStarted(set_number=3, lineup=list(LINEUP)))

    def test_lineup_validation(self) -> None:
        events = base_events()[:2]
        with pytest.raises(EngineError, match="distinct"):
            apply_strict(
                events, SetStarted(set_number=1, lineup=["p1", "p1", "p2", "p3", "p4", "p5"])
            )
        with pytest.raises(EngineError, match="not in roster"):
            apply_strict(
                events, SetStarted(set_number=1, lineup=["p1", "p2", "p3", "p4", "p5", "nope"])
            )
        with pytest.raises(EngineError, match="cannot be in the lineup"):
            apply_strict(
                events,
                SetStarted(set_number=1, lineup=list(LINEUP), liberos=["p1"]),
            )


class TestManualOverrides:
    def test_score_adjustment_applies(self) -> None:
        events = [*base_events(), rally(Team.US), rally(Team.THEM)]
        state = replay([*events, ScoreAdjusted(them_delta=1, note="ref missed the touch")])
        current = state.current_set
        assert current is not None
        assert (current.us_points, current.them_points) == (1, 2)
        # Rally-derived analytics are untouched by manual adjustments.
        assert current.per_rotation[1].total == 2

    def test_adjustment_cannot_go_negative_or_noop(self) -> None:
        events = base_events()
        with pytest.raises(EngineError, match="negative"):
            apply_strict(events, ScoreAdjusted(us_delta=-1))
        with pytest.raises(EngineError, match="change at least one"):
            apply_strict(events, ScoreAdjusted())

    def test_adjustment_can_undecide_a_set(self) -> None:
        events = [*base_events(), *rallies(Team.US, 25)]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.decided is Team.US
        state = replay([*events, ScoreAdjusted(us_delta=-1, note="scorer error")])
        current = state.current_set
        assert current is not None
        assert current.decided is None


class TestRetraction:
    def test_retract_heals_score_and_rotation(self) -> None:
        events = [*base_events(we_serve_first=False), rally(Team.US)]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.our_rotation == 2

        bad_rally = events[-1]
        state = replay([*events, EventRetracted(target_event_id=bad_rally.event_id)])
        current = state.current_set
        assert current is not None
        assert current.our_rotation == 1
        assert (current.us_points, current.them_points) == (0, 0)
        assert current.per_rotation[1].total == 0

    def test_retractable_event_skips_retracted(self) -> None:
        events = [*base_events(), rally(Team.US), rally(Team.THEM)]
        target = retractable_event(events)
        assert target is events[-1]
        events.append(EventRetracted(target_event_id=events[-1].event_id))
        target = retractable_event(events)
        assert target is events[-3]  # the first rally

    def test_session_created_is_not_undoable(self) -> None:
        events = base_events()[:1]
        assert retractable_event(events) is None
        with pytest.raises(EngineError, match="session_created"):
            apply_strict(events, EventRetracted(target_event_id=events[0].event_id))

    def test_retract_unknown_target_rejected(self) -> None:
        with pytest.raises(EngineError, match="unknown target"):
            apply_strict(base_events(), EventRetracted(target_event_id="nope"))

    def test_retract_a_retraction_rejected(self) -> None:
        events = [*base_events(), rally(Team.US)]
        retraction = EventRetracted(target_event_id=events[-1].event_id)
        events.append(retraction)
        with pytest.raises(EngineError, match="retract a retraction"):
            apply_strict(events, EventRetracted(target_event_id=retraction.event_id))


class TestPersonnel:
    def test_substitution_swaps_slot_ownership(self) -> None:
        events = [*base_events(), SubMade(player_in="p7", player_out="p3")]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.slot_owner[2] == "p7"
        assert current.on_court[2] == "p7"
        assert current.subs_used == 1

    def test_substitution_validation(self) -> None:
        events = base_events()
        with pytest.raises(EngineError, match="not on court"):
            apply_strict(events, SubMade(player_in="p7", player_out="p8"))
        with pytest.raises(EngineError, match="already on court"):
            apply_strict(events, SubMade(player_in="p1", player_out="p2"))
        with pytest.raises(EngineError, match="libero cannot enter"):
            apply_strict(events, SubMade(player_in="p8", player_out="p2"))

    def test_libero_swap_changes_floor_not_slots(self) -> None:
        events = [*base_events(), LiberoSwap(player_in="p8", player_out="p5")]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.on_court[4] == "p8"
        assert current.slot_owner[4] == "p5"

        # Rotation carries the libero along physically, slots keep identity.
        events = [
            *base_events(we_serve_first=False),
            LiberoSwap(player_in="p8", player_out="p5"),
            rally(Team.US),
        ]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.slot_owner[3] == "p5"
        assert current.on_court[3] == "p8"

    def test_libero_returns(self) -> None:
        events = [
            *base_events(),
            LiberoSwap(player_in="p8", player_out="p5"),
            LiberoSwap(player_in="p5", player_out="p8"),
        ]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.on_court[4] == "p5"

    def test_libero_swap_requires_a_libero(self) -> None:
        with pytest.raises(EngineError, match="must involve a libero"):
            apply_strict(base_events(), LiberoSwap(player_in="p7", player_out="p5"))

    def test_timeouts_count(self) -> None:
        events = [*base_events(), TimeoutCalled(team=Team.US), TimeoutCalled(team=Team.US)]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.timeouts["us"] == 2


class TestRallyMarkers:
    def test_rally_started_sets_flag_and_rally_ended_clears(self) -> None:
        events = [*base_events(), RallyStarted()]
        state = replay(events)
        current = state.current_set
        assert current is not None
        assert current.rally_in_progress

        state = replay([*events, rally(Team.US)])
        current = state.current_set
        assert current is not None
        assert not current.rally_in_progress

    def test_double_rally_start_rejected(self) -> None:
        events = [*base_events(), RallyStarted()]
        with pytest.raises(EngineError, match="already in progress"):
            apply_strict(events, RallyStarted())


class TestLenientReplay:
    def test_invalid_history_becomes_warning_not_crash(self) -> None:
        events = base_events()[:2]  # session + roster, no set yet
        events.append(rally(Team.US))  # invalid: no set in progress
        events.append(SetStarted(set_number=1, lineup=list(LINEUP)))
        events.append(rally(Team.US))
        state = replay(events)
        assert len(state.warnings) == 1
        assert "no set in progress" in state.warnings[0]
        current = state.current_set
        assert current is not None
        assert (current.us_points, current.them_points) == (1, 0)
