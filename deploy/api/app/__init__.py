"""HotMama API service package."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from fastapi import FastAPI


def create_app(*args: Any, **kwargs: Any) -> "FastAPI":
    """Return a FastAPI application instance.

    The import is performed lazily so that optional dependencies of
    :mod:`deploy.api.app.main` (such as ``redis``) are only required when the
    API service itself is instantiated. This allows other packages, like the GUI,
    to reuse shared constants without needing the full API dependency stack.
    """

    from .main import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
