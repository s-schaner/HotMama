from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel


class AppContext(BaseModel):
    db_url: str
    sessions_dir: Path
    config: Dict[str, Any]


class HookResult(BaseModel):
    events: List[Dict[str, Any]] = []
    artifacts: List[str] = []


class Plugin(Protocol):
    name: str
    version: str
    provides_ui: bool

    def on_register(self, ctx: AppContext) -> None: ...

    def on_session_open(self, session_id: str) -> None: ...

    def on_session_close(self, session_id: str) -> None: ...

    def on_clip_ingested(self, session_id: str, clip_id: str, clip_path: str) -> HookResult: ...

    def on_events_parsed(self, session_id: str, clip_id: str, events: List[Dict[str, Any]]) -> HookResult: ...

    def get_ui_blocks(self): ...


class SafePluginWrapper:
    def __init__(self, plugin: Plugin, name: Optional[str] = None) -> None:
        self.plugin = plugin
        self.name = name or getattr(plugin, "name", plugin.__class__.__name__)

    def __getattr__(self, item: str):
        return getattr(self.plugin, item)
