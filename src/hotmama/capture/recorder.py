"""Segmented recorder: camera → rolling MP4 segments on the host clock.

Why segments: an MP4 being written is unreadable (no moov atom until close),
so the recorder rolls a new file every ``segment_seconds``. Closed segments
are immediately extractable — a tagged moment becomes a clip at most one
segment-length after it happened. Segment metadata (wall-clock start/end,
frames, fps) is journaled to ``segments.json`` atomically on every roll, so
clip extraction — even after a crash or restart — is pure clock arithmetic.

Events are stamped by this same host at append time (see SessionManager),
so event time and recording time share one clock by construction.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hotmama.core.ids import utcnow

from .sources import CaptureSource, SourceSpec

LOGGER = logging.getLogger("hotmama.capture.recorder")

# Codec ladder: try browser-friendlier containers first, degrade to
# always-available ones. Chosen once per recording, reused for all segments.
_CODEC_LADDER: tuple[tuple[str, str], ...] = (
    ("avc1", ".mp4"),
    ("mp4v", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
)


@dataclass
class SegmentMeta:
    index: int
    path: str
    started_at: str
    ended_at: str | None = None
    frames: int = 0

    def start_dt(self) -> datetime:
        return datetime.fromisoformat(self.started_at)

    def end_dt(self) -> datetime | None:
        return datetime.fromisoformat(self.ended_at) if self.ended_at else None


@dataclass
class RecordingManifest:
    codec: str = ""
    container: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    finished: bool = False
    segments: list[SegmentMeta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "container": self.container,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "finished": self.finished,
            "segments": [vars(segment) for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordingManifest:
        manifest = cls(
            codec=str(data.get("codec", "")),
            container=str(data.get("container", "")),
            fps=float(data.get("fps", 0.0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            finished=bool(data.get("finished", False)),
        )
        manifest.segments = [
            SegmentMeta(**segment) for segment in data.get("segments", [])
        ]
        return manifest


def load_manifest(directory: Path) -> RecordingManifest | None:
    path = directory / "segments.json"
    if not path.exists():
        return None
    try:
        return RecordingManifest.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        LOGGER.warning("corrupt manifest at %s", path)
        return None


@dataclass
class RecorderStatus:
    state: str = "idle"  # idle | recording | finished | error
    error: str | None = None
    source: str = ""
    codec: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    started_at: str | None = None
    segments: int = 0
    frames_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class Recorder(threading.Thread):
    """One background thread: read frames, write segments, journal metadata."""

    def __init__(
        self,
        source: CaptureSource,
        out_dir: Path,
        *,
        segment_seconds: float = 60.0,
    ) -> None:
        super().__init__(name=f"recorder-{out_dir.name}", daemon=True)
        self._source = source
        self._dir = out_dir
        self._segment_seconds = max(1.0, segment_seconds)
        self._stop_flag = threading.Event()
        self._status_lock = threading.Lock()
        self._status = RecorderStatus(state="recording", source=source.spec.description)
        self._manifest = RecordingManifest(fps=source.fps)
        self._writer: Any = None
        self._segment_started: datetime | None = None
        self._segment_frames = 0
        self._cv2: Any = None
        self._latest_jpeg: bytes | None = None

    # -- public ------------------------------------------------------------

    def status(self) -> RecorderStatus:
        with self._status_lock:
            return RecorderStatus(**vars(self._status))

    def latest_jpeg(self) -> bytes | None:
        """The most recent frame as JPEG — the calibration UI taps on this."""
        with self._status_lock:
            return self._latest_jpeg

    def stop(self) -> None:
        self._stop_flag.set()

    # -- thread body ---------------------------------------------------------

    def run(self) -> None:
        import cv2

        self._cv2 = cv2
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._record_loop()
            self._finalize(state="finished")
        except Exception as err:  # noqa: BLE001 - the thread must not die silently
            LOGGER.exception("recorder failed")
            self._finalize(state="error", error=str(err))
        finally:
            self._source.release()

    def _record_loop(self) -> None:
        while not self._stop_flag.is_set():
            frame = self._source.read()
            if frame is None:
                return  # source exhausted (file) or persistently failing (stream)
            now = utcnow()
            if self._writer is None or self._segment_expired(now):
                self._roll_segment(frame, now)
            self._writer.write(frame)
            self._segment_frames += 1
            with self._status_lock:
                self._status.frames_total += 1
                refresh_snapshot = self._status.frames_total % 15 == 1
            if refresh_snapshot:
                ok, buffer = self._cv2.imencode(
                    ".jpg", frame, [int(self._cv2.IMWRITE_JPEG_QUALITY), 85]
                )
                if ok:
                    with self._status_lock:
                        self._latest_jpeg = buffer.tobytes()

    def _segment_expired(self, now: datetime) -> bool:
        assert self._segment_started is not None
        return (now - self._segment_started).total_seconds() >= self._segment_seconds

    def _roll_segment(self, frame: Any, now: datetime) -> None:
        self._close_segment(now)
        height, width = frame.shape[:2]
        index = len(self._manifest.segments) + 1

        if not self._manifest.codec:
            self._writer, codec, container = self._open_writer(index, width, height)
            self._manifest.codec = codec
            self._manifest.container = container
            self._manifest.width = width
            self._manifest.height = height
            with self._status_lock:
                self._status.codec = codec
                self._status.width = width
                self._status.height = height
                self._status.fps = self._manifest.fps
                self._status.started_at = now.isoformat()
        else:
            path = self._segment_path(index, self._manifest.container)
            self._writer = self._new_writer(
                path, self._manifest.codec, width, height
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"could not reopen writer with {self._manifest.codec}")

        self._segment_started = now
        self._segment_frames = 0
        self._manifest.segments.append(
            SegmentMeta(
                index=index,
                path=self._segment_path(index, self._manifest.container).name,
                started_at=now.isoformat(),
            )
        )
        self._write_manifest()
        with self._status_lock:
            self._status.segments = index

    def _close_segment(self, now: datetime) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        current = self._manifest.segments[-1]
        current.ended_at = now.isoformat()
        current.frames = self._segment_frames
        self._write_manifest()

    def _finalize(self, *, state: str, error: str | None = None) -> None:
        self._close_segment(utcnow())
        self._manifest.finished = True
        self._write_manifest()
        with self._status_lock:
            self._status.state = state
            self._status.error = error

    # -- writers & manifest ----------------------------------------------------

    def _segment_path(self, index: int, container: str) -> Path:
        return self._dir / f"seg_{index:04d}{container}"

    def _new_writer(self, path: Path, codec: str, width: int, height: int) -> Any:
        fourcc = self._cv2.VideoWriter_fourcc(*codec)
        return self._cv2.VideoWriter(str(path), fourcc, self._manifest.fps, (width, height))

    def _open_writer(self, index: int, width: int, height: int) -> tuple[Any, str, str]:
        for codec, container in _CODEC_LADDER:
            path = self._segment_path(index, container)
            writer = self._new_writer(path, codec, width, height)
            if writer.isOpened():
                return writer, codec, container
            writer.release()
            path.unlink(missing_ok=True)
        raise RuntimeError("no working video codec found (tried avc1/mp4v/XVID/MJPG)")

    def _write_manifest(self) -> None:
        tmp = self._dir / "segments.json.tmp"
        tmp.write_text(json.dumps(self._manifest.to_dict(), indent=1))
        tmp.replace(self._dir / "segments.json")


def open_recorder(
    spec: SourceSpec, out_dir: Path, *, segment_seconds: float
) -> Recorder:
    source = CaptureSource(spec)
    return Recorder(source, out_dir, segment_seconds=segment_seconds)
