"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hotmama import __version__

from .api import build_router
from .config import Settings
from .sessions import SessionManager
from .store import EventStore
from .ws import Hub


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = EventStore(settings.db_path)
    hub = Hub()
    manager = SessionManager(store, broadcast=hub.broadcast)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="HotMama v2", version=__version__, lifespan=lifespan)
    # LAN-only deployment: the PWA may be served from a dev port or the host.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_router(store, manager, hub))

    ui_dist = settings.resolve_ui_dist()
    if ui_dist is not None:
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")

    app.state.store = store
    app.state.manager = manager
    app.state.hub = hub
    return app
