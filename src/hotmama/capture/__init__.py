"""Video capture: camera → segmented recording → event-driven clips."""

from .service import (
    CaptureConflictError,
    CaptureService,
    CaptureUnavailableError,
    ClipOptions,
    SourceOpenError,
)

__all__ = [
    "CaptureConflictError",
    "CaptureService",
    "CaptureUnavailableError",
    "ClipOptions",
    "SourceOpenError",
]
