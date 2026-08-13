"""REST + WebSocket API.

Wire contract, kept deliberately small:

- ``POST /api/sessions`` — create a session (writes session_created and,
  when a roster is supplied, roster_registered).
- ``GET  /api/sessions`` — list sessions for the join screen.
- ``GET  /api/sessions/{id}`` — full snapshot (state + analytics summary).
- ``GET  /api/sessions/{id}/events`` — the raw log (audit, export, sync).
- ``POST /api/sessions/{id}/events`` — append one domain event.
- ``POST /api/sessions/{id}/undo`` — retract the latest undoable event.
- ``WS   /ws/sessions/{id}`` — snapshot on connect, every append broadcast,
  and appends accepted over the socket for low-latency statting.

Error mapping: malformed event → 422, rule violation (EngineError) → 409,
unknown session → 404.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hotmama.capture import (
    CaptureConflictError,
    CaptureService,
    CaptureUnavailableError,
    SourceOpenError,
)
from hotmama.core import RosterPlayer, dump_event
from hotmama.core.events import RosterRegistered, SessionCreated, SessionKind
from hotmama.core.ids import new_session_id

from .sessions import EngineError, NothingToUndoError, SessionManager
from .store import EventStore, UnknownSessionError
from .ws import Hub


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ""
    kind: SessionKind = SessionKind.MATCH
    our_team: str = "Us"
    opponent: str = "Them"
    best_of: int = Field(default=5, ge=1, le=5)
    set_points: int = Field(default=25, ge=1, le=99)
    final_set_points: int = Field(default=15, ge=1, le=99)
    roster: list[RosterPlayer] = Field(default_factory=list)
    actor: str | None = None


class AppendEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: dict[str, Any]
    actor: str | None = None


class UndoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str | None = None


class StartCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    rally_clips: bool = True
    tag_clips: bool = True


def build_router(
    store: EventStore,
    manager: SessionManager,
    hub: Hub,
    capture: CaptureService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/api/sessions", status_code=201)
    async def create_session(request: CreateSessionRequest) -> dict[str, Any]:
        if request.best_of not in (1, 3, 5):
            raise HTTPException(status_code=422, detail="best_of must be 1, 3, or 5")
        session_id = new_session_id()
        label = request.label or f"{request.our_team} vs {request.opponent}"
        store.create_session(session_id, kind=request.kind.value, label=label)
        created = SessionCreated(
            kind=request.kind,
            our_team=request.our_team,
            opponent=request.opponent,
            best_of=request.best_of,  # type: ignore[arg-type]
            set_points=request.set_points,
            final_set_points=request.final_set_points,
            actor=request.actor,
        )
        payload = await manager.append(session_id, dump_event(created))
        if request.roster:
            roster = RosterRegistered(players=request.roster, actor=request.actor)
            payload = await manager.append(session_id, dump_event(roster))
        return {"session_id": session_id, "label": label, **payload}

    @router.get("/api/sessions")
    async def list_sessions() -> list[dict[str, Any]]:
        return store.list_sessions()

    @router.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        try:
            return await manager.state_payload(session_id)
        except UnknownSessionError as err:
            raise HTTPException(status_code=404, detail="unknown session") from err

    @router.get("/api/sessions/{session_id}/events")
    async def get_events(session_id: str, after: int = 0) -> list[dict[str, Any]]:
        try:
            return store.load_events(session_id, after_seq=after)
        except UnknownSessionError as err:
            raise HTTPException(status_code=404, detail="unknown session") from err

    @router.post("/api/sessions/{session_id}/events")
    async def append_event(session_id: str, request: AppendEventRequest) -> dict[str, Any]:
        return await _append(manager, session_id, request.event, request.actor)

    @router.post("/api/sessions/{session_id}/undo")
    async def undo(session_id: str, request: UndoRequest | None = None) -> dict[str, Any]:
        actor = request.actor if request is not None else None
        try:
            return await manager.undo(session_id, actor=actor)
        except UnknownSessionError as err:
            raise HTTPException(status_code=404, detail="unknown session") from err
        except NothingToUndoError as err:
            raise HTTPException(status_code=409, detail="nothing to undo") from err

    def _require_session(session_id: str) -> None:
        if not store.session_exists(session_id):
            raise HTTPException(status_code=404, detail="unknown session")

    @router.post("/api/sessions/{session_id}/capture")
    async def start_capture(session_id: str, request: StartCaptureRequest) -> dict[str, Any]:
        _require_session(session_id)
        try:
            return await capture.start(
                session_id,
                request.source,
                rally_clips=request.rally_clips,
                tag_clips=request.tag_clips,
            )
        except CaptureConflictError as err:
            raise HTTPException(status_code=409, detail=str(err)) from err
        except SourceOpenError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except CaptureUnavailableError as err:
            raise HTTPException(status_code=503, detail=str(err)) from err

    @router.delete("/api/sessions/{session_id}/capture")
    async def stop_capture(session_id: str) -> dict[str, Any]:
        _require_session(session_id)
        return await capture.stop(session_id)

    @router.get("/api/sessions/{session_id}/capture")
    async def capture_status(session_id: str) -> dict[str, Any]:
        _require_session(session_id)
        return capture.status(session_id)

    @router.get("/api/sessions/{session_id}/clips")
    async def list_clips(session_id: str) -> list[dict[str, Any]]:
        _require_session(session_id)
        return store.list_clips(session_id)

    @router.websocket("/ws/sessions/{session_id}")
    async def session_socket(websocket: WebSocket, session_id: str) -> None:
        try:
            snapshot = await manager.state_payload(session_id)
        except UnknownSessionError:
            await websocket.close(code=4404, reason="unknown session")
            return
        await websocket.accept()
        await hub.join(session_id, websocket)
        try:
            await websocket.send_json(snapshot)
            while True:
                message = await websocket.receive_json()
                await _handle_ws_message(manager, websocket, session_id, message)
        except WebSocketDisconnect:
            pass
        finally:
            await hub.leave(session_id, websocket)

    return router


async def _handle_ws_message(
    manager: SessionManager,
    websocket: WebSocket,
    session_id: str,
    message: dict[str, Any],
) -> None:
    kind = message.get("type")
    if kind == "ping":
        await websocket.send_json({"type": "pong"})
        return
    if kind == "append":
        event = message.get("event")
        if not isinstance(event, dict):
            await websocket.send_json({"type": "error", "detail": "append requires an event"})
            return
        try:
            await manager.append(session_id, event, actor=message.get("actor"))
        except ValidationError as err:
            await websocket.send_json(
                {"type": "error", "detail": "invalid event", "errors": err.errors()}
            )
        except EngineError as err:
            await websocket.send_json({"type": "error", "detail": str(err)})
        return
    if kind == "undo":
        try:
            await manager.undo(session_id, actor=message.get("actor"))
        except NothingToUndoError:
            await websocket.send_json({"type": "error", "detail": "nothing to undo"})
        return
    await websocket.send_json({"type": "error", "detail": f"unknown message type {kind!r}"})


async def _append(
    manager: SessionManager,
    session_id: str,
    event: dict[str, Any],
    actor: str | None,
) -> dict[str, Any]:
    try:
        return await manager.append(session_id, event, actor=actor)
    except UnknownSessionError as err:
        raise HTTPException(status_code=404, detail="unknown session") from err
    except ValidationError as err:
        raise HTTPException(status_code=422, detail=err.errors()) from err
    except EngineError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
