"""Session manager: the single writer path from command to broadcast.

Every append goes through one per-session async lock: parse → strict-validate
against the replayed state → persist → refresh cache → notify listeners.
A rejected event never touches the store, so replay stays clean by
construction and the lenient path in the engine exists only for
schema-drift archaeology, not for daily operation.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from hotmama.analytics import match_summary
from hotmama.core import (
    AnyEvent,
    EventRetracted,
    MatchState,
    apply_strict,
    dump_event,
    parse_event,
    replay,
    retractable_event,
)
from hotmama.core.engine import EngineError

from .store import EventStore

Broadcast = Callable[[str, dict[str, Any]], Awaitable[None]]


class NothingToUndoError(Exception):
    pass


class SessionManager:
    def __init__(self, store: EventStore, broadcast: Broadcast | None = None) -> None:
        self._store = store
        self._broadcast = broadcast
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._events: dict[str, list[AnyEvent]] = {}
        self._states: dict[str, MatchState] = {}

    def set_broadcast(self, broadcast: Broadcast) -> None:
        self._broadcast = broadcast

    async def _load(self, session_id: str) -> tuple[list[AnyEvent], MatchState]:
        if session_id not in self._events:
            raw = await asyncio.to_thread(self._store.load_events, session_id)
            for row in raw:
                row.pop("seq", None)
            events = [parse_event(row) for row in raw]
            self._events[session_id] = events
            self._states[session_id] = replay(events)
        return self._events[session_id], self._states[session_id]

    async def state_payload(self, session_id: str) -> dict[str, Any]:
        async with self._locks[session_id]:
            events, state = await self._load(session_id)
        return _payload(state, last_seq=len(events))

    async def append(
        self,
        session_id: str,
        raw_event: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Validate and persist one event; returns the broadcast payload.

        Raises pydantic.ValidationError for malformed events and
        EngineError for rule violations — callers map those to 422/409.
        """
        if actor is not None and "actor" not in raw_event:
            raw_event = {**raw_event, "actor": actor}
        event = parse_event(raw_event)
        async with self._locks[session_id]:
            events, _ = await self._load(session_id)
            state = apply_strict(events, event)
            seq = await asyncio.to_thread(
                self._store.append_event, session_id, dump_event(event)
            )
            events.append(event)
            self._states[session_id] = state
            if _is_session_closed(event):
                await asyncio.to_thread(self._store.mark_closed, session_id)
            payload = _payload(state, last_seq=seq, event=event, seq=seq)
        if self._broadcast is not None:
            await self._broadcast(session_id, payload)
        return payload

    async def undo(self, session_id: str, *, actor: str | None = None) -> dict[str, Any]:
        """Retract the most recent undoable event (append a tombstone)."""
        async with self._locks[session_id]:
            events, _ = await self._load(session_id)
            target = retractable_event(events)
        if target is None:
            raise NothingToUndoError
        retraction = EventRetracted(target_event_id=target.event_id, actor=actor)
        payload = await self.append(session_id, dump_event(retraction))
        payload["retracted_event_id"] = target.event_id
        return payload


def _is_session_closed(event: AnyEvent) -> bool:
    return event.type == "session_closed"


def _payload(
    state: MatchState,
    *,
    last_seq: int,
    event: AnyEvent | None = None,
    seq: int | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "snapshot" if event is None else "event",
        "last_seq": last_seq,
        "state": state.to_public_dict(),
        "summary": match_summary(state),
    }
    if event is not None:
        body["event"] = dump_event(event)
        body["seq"] = seq
    return body


__all__ = ["EngineError", "NothingToUndoError", "SessionManager"]
