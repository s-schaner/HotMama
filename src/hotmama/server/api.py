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

import asyncio
from typing import Any

from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hotmama.capture import (
    CaptureConflictError,
    CaptureService,
    CaptureUnavailableError,
    SourceOpenError,
)
from hotmama.core import RosterPlayer, dump_event
from hotmama.core.events import (
    CvObservation,
    Producer,
    RosterRegistered,
    SessionCreated,
    SessionKind,
)
from hotmama.core.ids import new_session_id, utcnow
from hotmama.reports import (
    ReportUnavailableError,
    render_report_html,
    render_report_pdf,
)

from .config import Settings
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


class WorkerLeaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker: str = Field(min_length=1, max_length=80)


class WorkerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=80)
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    proposal: dict[str, Any] | None = None
    """Optional domain-event dict this observation proposes (confirm-flow)."""


class WorkerCompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    session_id: str
    worker: str = Field(min_length=1, max_length=80)
    ok: bool = True
    error: str | None = None
    producer: Producer = Producer.CV_WELL
    observations: list[WorkerObservation] = Field(default_factory=list, max_length=200)


def build_router(
    store: EventStore,
    manager: SessionManager,
    hub: Hub,
    capture: CaptureService,
    settings: Settings,
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

    # -- remote pull-worker feed (D15) ------------------------------------------

    def _check_worker_auth(authorization: str | None) -> None:
        if settings.worker_token is None:
            raise HTTPException(
                status_code=503,
                detail="worker feed disabled — set HOTMAMA_WORKER_TOKEN on the host",
            )
        if authorization != f"Bearer {settings.worker_token}":
            raise HTTPException(status_code=401, detail="invalid worker token")

    @router.post("/api/worker/lease")
    async def worker_lease(
        request: WorkerLeaseRequest,
        authorization: str | None = Header(default=None),
    ) -> Response:
        _check_worker_auth(authorization)
        job = store.lease_next_analysis(request.worker, settings.worker_lease_seconds)
        if job is None:
            return Response(status_code=204)
        session_row = store.get_session(str(job["session_id"])) or {}
        return JSONResponse(
            {
                "clip_id": job["clip_id"],
                "session_id": job["session_id"],
                "kind": job["kind"],
                "label": job["label"],
                "clip_url": job["url"],
                "start_at": job["start_at"],
                "end_at": job["end_at"],
                "attempts": job["attempts"],
                "session": {
                    "label": session_row.get("label", ""),
                    "kind": session_row.get("kind", ""),
                },
            }
        )

    @router.post("/api/worker/complete")
    async def worker_complete(
        request: WorkerCompleteRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_worker_auth(authorization)
        if request.producer not in (Producer.CV_WELL, Producer.CV_CLOUD):
            raise HTTPException(status_code=422, detail="producer must be a CV producer")
        if not store.analysis_job_exists(request.clip_id):
            raise HTTPException(status_code=404, detail="unknown analysis job")
        if not request.ok:
            store.complete_analysis(
                request.clip_id, ok=False, error=request.error or "worker error"
            )
            return {"status": "requeued_or_failed", "appended": 0}

        appended = 0
        auto_committed = 0
        for observation in request.observations:
            event = CvObservation(
                kind=observation.kind,
                data={**observation.data, "clip_id": request.clip_id},
                confidence=observation.confidence,
                producer=request.producer,
                actor=request.worker,
                proposal=observation.proposal,
            )
            await _append(manager, request.session_id, dump_event(event), request.worker)
            appended += 1
            threshold = settings.auto_commit_confidence
            if (
                observation.proposal is not None
                and threshold is not None
                and observation.confidence >= threshold
            ):
                proposed = _proposal_to_event(
                    observation.proposal,
                    source_event_id=event.event_id,
                    producer=request.producer.value,
                    confidence=observation.confidence,
                    occurred_at=event.occurred_at.isoformat(),
                    actor=request.worker,
                )
                try:
                    await manager.append(
                        request.session_id, proposed, actor=request.worker
                    )
                    auto_committed += 1
                except (ValidationError, EngineError):
                    # Invalid or rule-breaking proposal stays pending for a human.
                    pass
        store.complete_analysis(request.clip_id, ok=True)
        return {"status": "done", "appended": appended, "auto_committed": auto_committed}

    @router.get("/api/sessions/{session_id}/analysis")
    async def analysis_overview(session_id: str) -> dict[str, int]:
        _require_session(session_id)
        return store.analysis_overview(session_id)

    # -- confirm-flow: CV proposals need a human verdict ------------------------

    async def _pending_proposal(session_id: str, observation_id: str) -> dict[str, Any]:
        try:
            payload = await manager.state_payload(session_id)
        except UnknownSessionError as err:
            raise HTTPException(status_code=404, detail="unknown session") from err
        for proposal in payload["state"]["proposals"]:
            if proposal["event_id"] == observation_id:
                return dict(proposal)
        raise HTTPException(status_code=404, detail="no such pending proposal")

    @router.post("/api/sessions/{session_id}/proposals/{observation_id}/confirm")
    async def confirm_proposal(
        session_id: str,
        observation_id: str,
        request: UndoRequest | None = None,
    ) -> dict[str, Any]:
        proposal = await _pending_proposal(session_id, observation_id)
        event_dict = _proposal_to_event(
            proposal["proposal"],
            source_event_id=observation_id,
            producer=str(proposal["producer"]),
            confidence=float(proposal["confidence"]),
            occurred_at=str(proposal["occurred_at"]),
            actor=request.actor if request else None,
        )
        return await _append(
            manager, session_id, event_dict, request.actor if request else None
        )

    @router.post("/api/sessions/{session_id}/proposals/{observation_id}/dismiss")
    async def dismiss_proposal(
        session_id: str,
        observation_id: str,
        request: UndoRequest | None = None,
    ) -> dict[str, Any]:
        await _pending_proposal(session_id, observation_id)
        retraction = {
            "type": "event_retracted",
            "target_event_id": observation_id,
        }
        return await _append(
            manager, session_id, retraction, request.actor if request else None
        )

    async def _report_html(session_id: str) -> str:
        _require_session(session_id)
        payload = await manager.state_payload(session_id)
        row = store.get_session(session_id) or {}
        return render_report_html(
            state=payload["state"],
            summary=payload["summary"],
            clips=store.list_clips(session_id),
            label=str(row.get("label", "")),
            generated_at=utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

    @router.get("/api/sessions/{session_id}/report", response_class=HTMLResponse)
    async def report_html(session_id: str) -> HTMLResponse:
        return HTMLResponse(await _report_html(session_id))

    @router.get("/api/sessions/{session_id}/report.pdf")
    async def report_pdf(session_id: str) -> Response:
        html = await _report_html(session_id)
        try:
            pdf = await asyncio.to_thread(render_report_pdf, html)
        except ReportUnavailableError as err:
            raise HTTPException(status_code=503, detail=str(err)) from err
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="hotmama-{session_id}.pdf"'
                )
            },
        )

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


def _proposal_to_event(
    proposal: dict[str, Any],
    *,
    source_event_id: str,
    producer: str,
    confidence: float,
    occurred_at: str,
    actor: str | None,
) -> dict[str, Any]:
    """Turn a proposal dict into an appendable event, controlling provenance.

    The proposal's own identity fields are stripped so the confirmed event
    gets a fresh id and an explicit link back to the observation.
    """
    event = dict(proposal)
    event.pop("event_id", None)
    event.pop("source_event_id", None)
    event.pop("actor", None)
    event["source_event_id"] = source_event_id
    event.setdefault("producer", producer)
    event.setdefault("confidence", confidence)
    event.setdefault("occurred_at", occurred_at)
    if actor is not None:
        event["actor"] = actor
    return event


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
