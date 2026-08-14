"""Capture sources (D9): USB device index, RTSP/HTTP stream, or file.

A source spec is a plain string the coach can type:

- ``"0"``, ``"1"`` …            → local device index (USB webcam / capture card)
- ``"rtsp://…"``, ``"http…"``   → network stream (IP cam, phone-as-camera)
- anything else                 → a video file path (demos, film review)

File sources are paced to their native fps so a recorded file behaves like a
live camera — the wall-clock → media mapping that clip extraction relies on
stays valid for every source kind.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CaptureUnavailableError(RuntimeError):
    """OpenCV is not installed — capture features are disabled."""


class SourceOpenError(RuntimeError):
    """The capture source could not be opened."""


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as err:  # pragma: no cover - exercised only without extra
        raise CaptureUnavailableError(
            "opencv is required for capture — install hotmama[capture]"
        ) from err
    return cv2


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # "device" | "stream" | "file"
    target: str

    @property
    def description(self) -> str:
        return f"{self.kind}:{self.target}"


def parse_source_spec(spec: str) -> SourceSpec:
    spec = spec.strip()
    if not spec:
        raise SourceOpenError("empty capture source")
    if spec.isdigit():
        return SourceSpec(kind="device", target=spec)
    lowered = spec.lower()
    if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
        return SourceSpec(kind="stream", target=spec)
    return SourceSpec(kind="file", target=spec)


class CaptureSource:
    """Thin wrapper over cv2.VideoCapture with per-kind read semantics.

    ``read()`` returns the next frame or None when the source is exhausted
    (files) or has failed persistently (devices/streams).
    """

    _STREAM_RETRIES = 30

    def __init__(self, spec: SourceSpec) -> None:
        self.spec = spec
        cv2 = _cv2()
        if spec.kind == "device":
            self._capture = cv2.VideoCapture(int(spec.target))
        else:
            if spec.kind == "file" and not Path(spec.target).exists():
                raise SourceOpenError(f"video file not found: {spec.target}")
            self._capture = cv2.VideoCapture(spec.target)
        if not self._capture.isOpened():
            self._capture.release()
            raise SourceOpenError(f"could not open capture source {spec.description}")

        raw_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps: float = raw_fps if 1.0 <= raw_fps <= 240.0 else 30.0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._paced = spec.kind == "file"
        self._next_deadline = time.monotonic()
        self._misses = 0

    def read(self) -> Any | None:
        if self._paced:
            now = time.monotonic()
            if now < self._next_deadline:
                time.sleep(self._next_deadline - now)
            self._next_deadline = max(self._next_deadline, now) + 1.0 / self.fps

        while True:
            success, frame = self._capture.read()
            if success:
                self._misses = 0
                return frame
            if self.spec.kind == "file":
                return None
            self._misses += 1
            if self._misses >= self._STREAM_RETRIES:
                return None
            time.sleep(0.1)

    def release(self) -> None:
        self._capture.release()
