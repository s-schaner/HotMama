"""The reducer: an append-only event log in, derived match state out.

Two entry points with different postures:

- :func:`apply_strict` — raises :class:`EngineError` on any rule violation.
  The server runs this *before* persisting a new event, so bad commands are
  rejected at the door.
- :func:`replay` — never raises. A stored log is history; if an event in it
  is invalid (schema drift, a buggy old producer), it is skipped and the
  problem is recorded in ``state.warnings``. Replay must always produce a
  usable state from any log that was once accepted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .events import (
    AnyEvent,
    CvObservation,
    EventRetracted,
    LiberoSwap,
    MomentTagged,
    NoteAdded,
    RallyEnded,
    RallyStarted,
    RosterRegistered,
    ScoreAdjusted,
    SessionClosed,
    SessionCreated,
    SetEnded,
    SetStarted,
    SubMade,
    Team,
    TimeoutCalled,
)
from .state import MatchState, PointRecord, SetState, TagRecord


class EngineError(ValueError):
    """A domain rule was violated. Raised only under strict application."""


def replay(events: Sequence[AnyEvent]) -> MatchState:
    """Fold the full log into a state, honoring retractions, never raising."""
    retracted = {
        event.target_event_id for event in events if isinstance(event, EventRetracted)
    }
    state = MatchState()
    for event in events:
        if isinstance(event, EventRetracted) or event.event_id in retracted:
            continue
        try:
            _apply(state, event)
        except EngineError as err:
            state.warnings.append(f"skipped {event.type} ({event.event_id}): {err}")
        else:
            state.applied_events += 1
            state.last_event_id = event.event_id
    return state


def apply_strict(events: Sequence[AnyEvent], new_event: AnyEvent) -> MatchState:
    """Validate ``new_event`` against the state produced by ``events``.

    Returns the state with the new event applied. The existing log is replayed
    leniently (it is history); only the new event is held to strict rules.
    """
    if isinstance(new_event, EventRetracted):
        _validate_retraction(events, new_event)
        return replay([*events, new_event])
    state = replay(events)
    _apply(state, new_event)
    state.applied_events += 1
    state.last_event_id = new_event.event_id
    return state


def retractable_event(events: Sequence[AnyEvent]) -> AnyEvent | None:
    """The most recent event an "undo" should target, or None."""
    retracted = {
        event.target_event_id for event in events if isinstance(event, EventRetracted)
    }
    for event in reversed(events):
        if isinstance(event, EventRetracted) or event.event_id in retracted:
            continue
        if isinstance(event, SessionCreated):
            return None
        return event
    return None


def _validate_retraction(events: Sequence[AnyEvent], retraction: EventRetracted) -> None:
    target = retraction.target_event_id
    for event in events:
        if event.event_id == target:
            if isinstance(event, SessionCreated):
                raise EngineError("cannot retract session_created")
            if isinstance(event, EventRetracted):
                raise EngineError("cannot retract a retraction")
            return
    raise EngineError(f"unknown target event {target!r}")


def _apply(state: MatchState, event: AnyEvent) -> None:
    if isinstance(event, SessionCreated):
        _apply_session_created(state, event)
    elif isinstance(event, RosterRegistered):
        _apply_roster(state, event)
    elif isinstance(event, SetStarted):
        _apply_set_started(state, event)
    elif isinstance(event, RallyStarted):
        _apply_rally_started(state, event)
    elif isinstance(event, RallyEnded):
        _apply_rally_ended(state, event)
    elif isinstance(event, ScoreAdjusted):
        _apply_score_adjusted(state, event)
    elif isinstance(event, MomentTagged):
        _apply_moment_tagged(state, event)
    elif isinstance(event, SubMade):
        _apply_sub(state, event)
    elif isinstance(event, LiberoSwap):
        _apply_libero_swap(state, event)
    elif isinstance(event, TimeoutCalled):
        _apply_timeout(state, event)
    elif isinstance(event, SetEnded):
        _apply_set_ended(state, event)
    elif isinstance(event, SessionClosed):
        _require(state.created, "session not created")
        state.session_closed = True
    elif isinstance(event, NoteAdded):
        _require(state.created, "session not created")
        state.notes.append(
            {"event_id": event.event_id, "text": event.text, "at": event.occurred_at.isoformat()}
        )
    elif isinstance(event, CvObservation):
        state.cv_observations += 1
    else:  # pragma: no cover - exhaustiveness guard for future event types
        raise EngineError(f"unhandled event type {type(event).__name__}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EngineError(message)


def _open_set(state: MatchState) -> SetState:
    current = state.current_set
    _require(current is not None, "no set in progress")
    assert current is not None
    return current


def _apply_session_created(state: MatchState, event: SessionCreated) -> None:
    _require(not state.created, "session already created")
    state.created = True
    state.kind = event.kind
    state.our_team = event.our_team
    state.opponent = event.opponent
    state.best_of = event.best_of
    state.set_points = event.set_points
    state.final_set_points = event.final_set_points


def _apply_roster(state: MatchState, event: RosterRegistered) -> None:
    _require(state.created, "session not created")
    seen: set[str] = set()
    jerseys: set[int] = set()
    for player in event.players:
        _require(player.player_id not in seen, f"duplicate player_id {player.player_id!r}")
        seen.add(player.player_id)
        if player.jersey is not None:
            _require(player.jersey not in jerseys, f"duplicate jersey {player.jersey}")
            jerseys.add(player.jersey)
    state.roster = {player.player_id: player for player in event.players}


def _apply_set_started(state: MatchState, event: SetStarted) -> None:
    _require(state.created, "session not created")
    _require(not state.session_closed, "session is closed")
    _require(state.current_set is None, "a set is already in progress")
    _require(not state.match_over, "match is already decided")
    expected = len(state.sets) + 1
    _require(
        event.set_number == expected,
        f"expected set {expected}, got {event.set_number}",
    )
    _require(event.set_number <= state.best_of, "set number exceeds match format")

    lineup = list(event.lineup)
    _require(len(set(lineup)) == 6, "lineup must contain 6 distinct players")
    for player_id in lineup:
        _require(player_id in state.roster, f"lineup player {player_id!r} not in roster")
    for libero in event.liberos:
        _require(libero in state.roster, f"libero {libero!r} not in roster")
        _require(libero not in lineup, f"libero {libero!r} cannot be in the lineup")

    deciding = state.best_of > 1 and event.set_number == state.best_of
    state.sets.append(
        SetState(
            set_number=event.set_number,
            to_win=state.final_set_points if deciding else state.set_points,
            serving=Team.US if event.we_serve_first else Team.THEM,
            slot_owner=lineup,
            on_court=list(lineup),
            liberos=list(event.liberos),
        )
    )


def _apply_rally_started(state: MatchState, event: RallyStarted) -> None:
    current = _open_set(state)
    _require(current.decided is None, "set is decided — end it or adjust the score")
    _require(not current.rally_in_progress, "rally already in progress")
    current.rally_in_progress = True
    current.rally_started_at = event.occurred_at


def _apply_rally_ended(state: MatchState, event: RallyEnded) -> None:
    current = _open_set(state)
    _require(current.decided is None, "set is decided — end it or adjust the score")
    if event.player_id is not None:
        _require(event.player_id in state.roster, f"player {event.player_id!r} not in roster")

    served_by = current.serving
    rotation = current.our_rotation
    tally = current.per_rotation[rotation]
    we_won = event.winner is Team.US

    if event.winner is Team.US:
        current.us_points += 1
    else:
        current.them_points += 1

    if served_by is Team.US:
        if we_won:
            tally.serve_won += 1
        else:
            tally.serve_lost += 1
    elif we_won:
        tally.recv_won += 1
    else:
        tally.recv_lost += 1

    reasons = tally.won_reasons if we_won else tally.lost_reasons
    reasons[event.reason.value] = reasons.get(event.reason.value, 0) + 1

    current.points.append(
        PointRecord(
            number=len(current.points) + 1,
            winner=event.winner,
            reason=event.reason,
            served_by=served_by,
            our_rotation=rotation,
            our_server=current.our_server if served_by is Team.US else None,
            player_id=event.player_id,
            opponent_jersey=event.opponent_jersey,
            us_points=current.us_points,
            them_points=current.them_points,
            event_id=event.event_id,
            occurred_at=event.occurred_at,
        )
    )

    # Side-out for us: we won a rally we were receiving → rotate, then serve.
    if we_won and served_by is Team.THEM:
        current.rotate()
    current.serving = event.winner
    current.rally_in_progress = False
    current.rally_started_at = None


def _apply_score_adjusted(state: MatchState, event: ScoreAdjusted) -> None:
    current = _open_set(state)
    _require(
        event.us_delta != 0 or event.them_delta != 0,
        "adjustment must change at least one score",
    )
    us = current.us_points + event.us_delta
    them = current.them_points + event.them_delta
    _require(us >= 0 and them >= 0, "score cannot go negative")
    current.us_points = us
    current.them_points = them


def _apply_moment_tagged(state: MatchState, event: MomentTagged) -> None:
    _require(state.created, "session not created")
    if event.player_id is not None:
        _require(event.player_id in state.roster, f"player {event.player_id!r} not in roster")
    current = state.sets[-1] if state.sets else None
    state.tags.append(
        TagRecord(
            event_id=event.event_id,
            tag=event.tag,
            player_id=event.player_id,
            note=event.note,
            custom_label=event.custom_label,
            set_number=current.set_number if current else 0,
            us_points=current.us_points if current else 0,
            them_points=current.them_points if current else 0,
            occurred_at=event.occurred_at,
        )
    )


def _apply_sub(state: MatchState, event: SubMade) -> None:
    current = _open_set(state)
    _require(event.player_in in state.roster, f"player {event.player_in!r} not in roster")
    _require(event.player_out in current.slot_owner, f"{event.player_out!r} is not on court")
    _require(event.player_in not in current.slot_owner, f"{event.player_in!r} already on court")
    _require(event.player_in not in current.liberos, "libero cannot enter via substitution")
    slot = current.slot_owner.index(event.player_out)
    current.slot_owner[slot] = event.player_in
    if event.player_out in current.on_court:
        current.on_court[current.on_court.index(event.player_out)] = event.player_in
    current.subs_used += 1


def _apply_libero_swap(state: MatchState, event: LiberoSwap) -> None:
    current = _open_set(state)
    going_on = event.player_in in current.liberos
    going_off = event.player_out in current.liberos
    _require(going_on or going_off, "libero swap must involve a libero")
    _require(not (going_on and going_off), "cannot swap a libero for a libero")
    _require(event.player_out in current.on_court, f"{event.player_out!r} is not on the floor")
    _require(event.player_in not in current.on_court, f"{event.player_in!r} already on the floor")
    if going_on:
        _require(
            event.player_in in state.roster and event.player_out in current.slot_owner,
            "libero must replace a rotation player",
        )
    else:
        _require(
            event.player_in in current.slot_owner,
            "returning player must own a rotation slot",
        )
    current.on_court[current.on_court.index(event.player_out)] = event.player_in


def _apply_timeout(state: MatchState, event: TimeoutCalled) -> None:
    current = _open_set(state)
    current.timeouts[event.team.value] = current.timeouts.get(event.team.value, 0) + 1


def _apply_set_ended(state: MatchState, event: SetEnded) -> None:
    current = _open_set(state)
    _require(
        current.us_points != current.them_points,
        "cannot end a tied set — adjust the score first",
    )
    current.finished = True
    current.won_by = Team.US if current.us_points > current.them_points else Team.THEM
    current.rally_in_progress = False
    current.rally_started_at = None


def events_from_dicts(rows: Iterable[dict[str, object]]) -> list[AnyEvent]:
    """Parse stored/wire rows into typed events (validation included)."""
    from .events import parse_event

    return [parse_event(dict(row)) for row in rows]
