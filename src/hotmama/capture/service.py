"""CaptureService: recording lifecycle + event-driven clip extraction.

The service observes the same event stream every UI device does. When a
``moment_tagged`` lands it schedules a human-grade clip around it; when a
``rally_ended`` lands it cuts the rally chunk that phase 4 will ship to the
Well (D8: rally-chunked near-live). Extraction waits until the recorder has
closed the segments covering the window — worst case one segment length —
then runs ffmpeg off the event loop.

Rally windows, best evidence first:
1. an explicit ``rally_started`` marker,
2. one second after the previous rally ended,
3. a fixed look-back, capped at ``rally_max_seconds``.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from hotmama.core.ids import utcnow

from .clips import ClipError, extract_clip, recorded_range
from .recorder import Recorder, RecordingManifest, load_manifest, open_recorder
from .sources import CaptureUnavailableError, SourceOpenError, parse_source_spec

LOGGER = logging.getLogger("hotmama.capture.service")

Broadcast = Callable[[str, dict[str, Any]], Awaitable[None]]


class ClipStore(Protocol):
    """The persistence the clip pipeline needs — implemented by the server store."""

    def insert_clip(
        self,
        clip_id: str,
        *,
        session_id: str,
        event_id: str,
        kind: str,
        label: str,
        start_at: str,
        end_at: str,
    ) -> None: ...

    def set_clip_ready(self, clip_id: str, *, path: str, url: str) -> None: ...

    def set_clip_failed(self, clip_id: str, *, error: str) -> None: ...

    def list_clips(self, session_id: str) -> list[dict[str, Any]]: ...

    def enqueue_analysis(self, clip_id: str, session_id: str) -> None: ...


class CaptureConflictError(RuntimeError):
    """Capture is already running for this session."""


class ClipOptions:
    def __init__(
        self,
        *,
        segment_seconds: float = 60.0,
        tag_pre_seconds: float = 8.0,
        tag_post_seconds: float = 4.0,
        rally_pad_seconds: float = 2.0,
        rally_max_seconds: float = 60.0,
    ) -> None:
        self.segment_seconds = segment_seconds
        self.tag_pre_seconds = tag_pre_seconds
        self.tag_post_seconds = tag_post_seconds
        self.rally_pad_seconds = rally_pad_seconds
        self.rally_max_seconds = rally_max_seconds


class _ActiveCapture:
    def __init__(self, recorder: Recorder, *, rally_clips: bool, tag_clips: bool) -> None:
        self.recorder = recorder
        self.rally_clips = rally_clips
        self.tag_clips = tag_clips


class CaptureService:
    def __init__(
        self,
        store: ClipStore,
        media_root: Path,
        *,
        options: ClipOptions | None = None,
        broadcast: Broadcast | None = None,
    ) -> None:
        self._store = store
        self._media_root = media_root
        self._options = options or ClipOptions()
        self._broadcast = broadcast
        self._active: dict[str, _ActiveCapture] = {}
        self._rally_started_at: dict[str, datetime] = {}
        self._last_rally_end: dict[str, datetime] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    # -- paths ---------------------------------------------------------------

    def recording_dir(self, session_id: str) -> Path:
        return self._media_root / session_id / "recording"

    def clips_dir(self, session_id: str) -> Path:
        return self._media_root / session_id / "clips"

    def media_url(self, session_id: str, path: Path) -> str:
        return f"/media/{session_id}/clips/{path.name}"

    # -- lifecycle -------------------------------------------------------------

    async def start(
        self,
        session_id: str,
        source_spec: str,
        *,
        rally_clips: bool = True,
        tag_clips: bool = True,
    ) -> dict[str, Any]:
        existing = self._active.get(session_id)
        if existing and existing.recorder.status().state == "recording":
            raise CaptureConflictError("capture already running for this session")

        spec = parse_source_spec(source_spec)
        recorder = await asyncio.to_thread(
            open_recorder,
            spec,
            self.recording_dir(session_id),
            segment_seconds=self._options.segment_seconds,
        )
        recorder.start()
        self._active[session_id] = _ActiveCapture(
            recorder, rally_clips=rally_clips, tag_clips=tag_clips
        )
        status = self.status(session_id)
        await self._broadcast_capture(session_id, status)
        return status

    async def stop(self, session_id: str) -> dict[str, Any]:
        active = self._active.get(session_id)
        if active is not None:
            active.recorder.stop()
            await asyncio.to_thread(active.recorder.join, 10.0)
        status = self.status(session_id)
        await self._broadcast_capture(session_id, status)
        return status

    def status(self, session_id: str) -> dict[str, Any]:
        active = self._active.get(session_id)
        if active is None:
            manifest = load_manifest(self.recording_dir(session_id))
            state = "finished" if manifest and manifest.segments else "idle"
            return {"state": state}
        return active.recorder.status().to_dict()

    async def shutdown(self) -> None:
        for session_id in list(self._active):
            await self.stop(session_id)
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # -- event stream --------------------------------------------------------

    async def on_event(self, session_id: str, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        occurred_raw = event.get("occurred_at")
        if not isinstance(event_type, str) or not isinstance(occurred_raw, str):
            return
        try:
            occurred = datetime.fromisoformat(occurred_raw)
        except ValueError:
            return

        if event_type == "rally_started":
            self._rally_started_at[session_id] = occurred
            return
        if event_type == "session_closed":
            if session_id in self._active:
                await self.stop(session_id)
            return

        active = self._active.get(session_id)
        has_recording = active is not None or load_manifest(
            self.recording_dir(session_id)
        ) is not None
        if not has_recording:
            return

        if event_type == "moment_tagged" and (active is None or active.tag_clips):
            start = occurred - timedelta(seconds=self._options.tag_pre_seconds)
            end = occurred + timedelta(seconds=self._options.tag_post_seconds)
            label = str(event.get("tag", "moment"))
            self._schedule_clip(
                session_id,
                event_id=str(event.get("event_id", "")),
                kind="tag",
                label=label,
                start=start,
                end=end,
                reencode=True,
            )
            return

        if event_type == "rally_ended" and (active is None or active.rally_clips):
            pad = timedelta(seconds=self._options.rally_pad_seconds)
            end = occurred + pad
            start = self._rally_window_start(session_id, occurred) - pad
            self._rally_started_at.pop(session_id, None)
            self._last_rally_end[session_id] = occurred
            winner = event.get("winner", "")
            self._schedule_clip(
                session_id,
                event_id=str(event.get("event_id", "")),
                kind="rally",
                label=f"rally→{winner}",
                start=start,
                end=end,
                reencode=False,
            )

    def _rally_window_start(self, session_id: str, ended_at: datetime) -> datetime:
        cap = ended_at - timedelta(seconds=self._options.rally_max_seconds)
        marker = self._rally_started_at.get(session_id)
        if marker is not None and marker < ended_at:
            return max(marker, cap)
        previous_end = self._last_rally_end.get(session_id)
        if previous_end is not None and previous_end < ended_at:
            return max(previous_end + timedelta(seconds=1), cap)
        return cap

    # -- clip pipeline ---------------------------------------------------------

    def _schedule_clip(
        self,
        session_id: str,
        *,
        event_id: str,
        kind: str,
        label: str,
        start: datetime,
        end: datetime,
        reencode: bool,
    ) -> None:
        clip_id = f"c_{secrets.token_hex(6)}"
        self._store.insert_clip(
            clip_id,
            session_id=session_id,
            event_id=event_id,
            kind=kind,
            label=label,
            start_at=start.isoformat(),
            end_at=end.isoformat(),
        )
        task = asyncio.create_task(
            self._produce_clip(
                session_id,
                clip_id=clip_id,
                kind=kind,
                start=start,
                end=end,
                reencode=reencode,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _produce_clip(
        self,
        session_id: str,
        *,
        clip_id: str,
        kind: str,
        start: datetime,
        end: datetime,
        reencode: bool,
    ) -> None:
        try:
            manifest = await self._wait_for_window(session_id, end)
            suffix = ".mp4" if reencode else (manifest.container or ".mp4")
            out_path = self.clips_dir(session_id) / f"{clip_id}{suffix}"
            await asyncio.to_thread(
                extract_clip,
                manifest,
                self.recording_dir(session_id),
                start,
                end,
                out_path,
                reencode=reencode,
            )
            self._store.set_clip_ready(
                clip_id, path=str(out_path), url=self.media_url(session_id, out_path)
            )
            if kind == "rally":
                # Feed the pull-worker queue (D15): rally chunks are CV fodder.
                self._store.enqueue_analysis(clip_id, session_id)
        except (ClipError, TimeoutError) as err:
            self._store.set_clip_failed(clip_id, error=str(err))
        except Exception as err:  # noqa: BLE001 - never let a clip kill the loop
            LOGGER.exception("clip %s failed", clip_id)
            self._store.set_clip_failed(clip_id, error=f"internal: {err}")
        await self._broadcast_clips(session_id)

    async def _wait_for_window(self, session_id: str, end: datetime) -> RecordingManifest:
        """Wait until closed segments cover ``end`` or the recording finishes."""
        deadline = utcnow() + timedelta(seconds=self._options.segment_seconds * 2 + 15)
        while True:
            manifest = load_manifest(self.recording_dir(session_id))
            if manifest is not None:
                span = recorded_range(manifest)
                covered = span is not None and span[1] >= end
                if covered or manifest.finished:
                    return manifest
                active = self._active.get(session_id)
                if active is not None and active.recorder.status().state == "error":
                    return manifest
            if utcnow() >= deadline:
                if manifest is not None:
                    return manifest
                raise TimeoutError("no recording manifest appeared")
            await asyncio.sleep(0.5)

    # -- broadcasts --------------------------------------------------------------

    async def _broadcast_capture(self, session_id: str, status: dict[str, Any]) -> None:
        if self._broadcast is not None:
            await self._broadcast(session_id, {"type": "capture", "status": status})

    async def _broadcast_clips(self, session_id: str) -> None:
        if self._broadcast is not None:
            clips = self._store.list_clips(session_id)
            await self._broadcast(session_id, {"type": "clips", "clips": clips})


__all__ = [
    "CaptureConflictError",
    "CaptureService",
    "CaptureUnavailableError",
    "ClipOptions",
    "SourceOpenError",
]
