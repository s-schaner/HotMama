from __future__ import annotations

from collections import defaultdict
from typing import Callable, DefaultDict, List

EventHandler = Callable[..., None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: DefaultDict[str, List[EventHandler]] = defaultdict(list)

    def subscribe(self, event: str, handler: EventHandler) -> None:
        self._handlers[event].append(handler)

    def emit(self, event: str, *args, **kwargs) -> None:
        for handler in list(self._handlers.get(event, [])):
            try:
                handler(*args, **kwargs)
            except Exception:
                # Failing handlers are ignored but kept for debugging.
                continue
