"""HotMama worker service package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .worker import Worker

__all__ = ["Worker"]


def __getattr__(name: str) -> Any:  # pragma: no cover - dynamic import helper
    if name == "Worker":
        from .worker import Worker as _Worker

        return _Worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
