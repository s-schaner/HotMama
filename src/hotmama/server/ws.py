"""WebSocket hub: every device at the gym watches (and writes) one session."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

LOGGER = logging.getLogger("hotmama.server.ws")


class Hub:
    def __init__(self) -> None:
        self._clients: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def join(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients[session_id].add(websocket)

    async def leave(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients[session_id].discard(websocket)
            if not self._clients[session_id]:
                self._clients.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients.get(session_id, ()))
        for websocket in targets:
            try:
                await websocket.send_json(message)
            except Exception:  # noqa: BLE001 - a dead socket must never break the rest
                LOGGER.debug("dropping dead websocket for %s", session_id)
                await self.leave(session_id, websocket)
