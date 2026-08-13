"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hotmama import __version__
from hotmama.capture import CaptureService, ClipOptions

from .api import build_router
from .config import Settings
from .sessions import SessionManager
from .store import EventStore
from .ws import Hub


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = EventStore(settings.db_path)
    hub = Hub()
    capture = CaptureService(
        store,
        settings.media_root,
        options=ClipOptions(
            segment_seconds=settings.segment_seconds,
            tag_pre_seconds=settings.tag_pre_seconds,
            tag_post_seconds=settings.tag_post_seconds,
            rally_pad_seconds=settings.rally_pad_seconds,
            rally_max_seconds=settings.rally_max_seconds,
        ),
        broadcast=hub.broadcast,
    )

    async def fanout(session_id: str, payload: dict[str, Any]) -> None:
        await hub.broadcast(session_id, payload)
        event = payload.get("event")
        if isinstance(event, dict):
            await capture.on_event(session_id, event)

    manager = SessionManager(store, broadcast=fanout)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await capture.shutdown()
            store.close()

    app = FastAPI(title="HotMama v2", version=__version__, lifespan=lifespan)
    # LAN-only deployment: the PWA may be served from a dev port or the host.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_router(store, manager, hub, capture, settings))

    settings.media_root.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=settings.media_root), name="media")

    ui_dist = settings.resolve_ui_dist()
    if ui_dist is not None:
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")

    app.state.store = store
    app.state.manager = manager
    app.state.hub = hub
    app.state.capture = capture
    return app
