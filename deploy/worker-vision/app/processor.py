"""Vision processing pipeline stub."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import torch

from .config import Settings, get_settings
from .schemas import ProcessResult, QueuedJob

LOGGER = logging.getLogger("hotmama.worker.processor")

_TIME_PATTERN = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})(?:\.(?P<millis>\d{1,3}))?$"
)


def _parse_timecode(value: str) -> float:
    match = _TIME_PATTERN.fullmatch(value)
    if not match:
        raise ValueError("timecode must match HH:MM:SS(.mmm)")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("minutes and seconds must be less than 60")
    millis_group = match.group("millis")
    millis = int(millis_group) if millis_group else 0
    if millis_group and len(millis_group) < 3:
        millis *= 10 ** (3 - len(millis_group))
    total_millis = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
    return total_millis / 1000.0


def _format_timecode(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    hours, remainder = divmod(total_millis, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    if millis:
        millis_str = f"{millis:03d}".rstrip("0")
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis_str}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class VisionProcessor:
    """Performs lightweight processing to validate the pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: torch.nn.Module | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            LOGGER.info(
                "loading vision model stub",
                extra={"service": self._settings.service_name},
            )
            self._model = torch.nn.Sequential(torch.nn.Identity())

    def _summarize_video(self, source_uri: str) -> dict[str, Any]:
        capture = cv2.VideoCapture(source_uri)
        if not capture.isOpened():
            return {"frames": 0, "fps": 0.0, "valid": False}
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
        return {"frames": frame_count, "fps": fps, "valid": True}

    def process(self, job: QueuedJob) -> ProcessResult:
        self._ensure_model()
        payload = job.payload
        source_uri = payload.get("source_uri", "")
        summary = self._summarize_video(source_uri) if source_uri else {}

        artifact_dir = self._settings.artifact_root / str(job.job_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = artifact_dir / "result.json"

        tensor = torch.tensor([len(source_uri)], dtype=torch.float32)
        _ = self._model(tensor)

        overlays = self._normalise_overlays(payload)
        video_path: Path | None = None
        video_metadata: dict[str, Any] = {
            "source_uri": source_uri,
            "overlays_requested": [
                name for name, enabled in overlays.items() if enabled
            ],
            "frames": 0,
            "fps": 0.0,
        }
        render_error: str | None = None

        if source_uri:
            video_path, video_metadata, render_error = self._render_overlays(
                source_uri=source_uri,
                overlays=overlays,
                fps_override=self._extract_fps(payload),
                artifact_dir=artifact_dir,
                clips=payload.get("clips"),
            )
        else:
            render_error = "no source_uri provided"
            video_metadata["error"] = render_error

        metadata = {
            "job_id": str(job.job_id),
            "profile": job.profile,
            "summary": summary,
            "parameters": payload.get("options")
            if isinstance(payload.get("options"), dict)
            else payload.get("parameters", {}),
            "video": video_metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        if video_path is None:
            message = f"video rendering failed: {render_error}" if render_error else "video rendering failed"
            LOGGER.error(
                "video rendering failed",
                extra={
                    "service": self._settings.service_name,
                    "job_id": str(job.job_id),
                    "error": render_error,
                },
            )
            return ProcessResult(status="failed", message=message, artifact_path=None)

        message = "processing complete"
        if video_metadata.get("fallback_used"):
            message = (
                "processing complete (fallback: "
                f"{Path(str(video_metadata.get('path', ''))).name}"
                f" via {video_metadata.get('codec')})"
            )

        return ProcessResult(
            status="completed",
            message=message,
            artifact_path=str(video_path),
        )

    def _extract_fps(self, payload: dict[str, Any]) -> float | None:
        fps_value = payload.get("fps")
        if isinstance(fps_value, (int, float)) and fps_value > 0:
            return float(fps_value)
        options = payload.get("options")
        if isinstance(options, dict):
            candidate = options.get("fps")
            if isinstance(candidate, (int, float)) and candidate > 0:
                return float(candidate)
        return None

    def _resolve_clip_ranges(
        self, clips: Any, fps: float
    ) -> tuple[list[tuple[int, int]], list[dict[str, str]]]:
        if fps <= 0 or not isinstance(clips, list):
            return [], []

        entries: list[tuple[tuple[int, int], dict[str, str]]] = []
        for entry in clips:
            if not isinstance(entry, dict):
                continue
            start_value = entry.get("start")
            end_value = entry.get("end")
            if not isinstance(start_value, str) or not isinstance(end_value, str):
                continue
            start_text = start_value.strip()
            end_text = end_value.strip()
            if not start_text or not end_text:
                continue
            try:
                start_seconds = _parse_timecode(start_text)
                end_seconds = _parse_timecode(end_text)
            except ValueError:
                continue
            if end_seconds <= start_seconds:
                continue
            start_frame = max(0, int(math.floor(start_seconds * fps)))
            end_frame = max(start_frame + 1, int(math.ceil(end_seconds * fps)))
            entries.append(
                (
                    (start_frame, end_frame),
                    {
                        "start": _format_timecode(start_seconds),
                        "end": _format_timecode(end_seconds),
                    },
                )
            )

        if not entries:
            return [], []

        entries.sort(key=lambda item: item[0][0])
        ranges = [item[0] for item in entries]
        metadata = [item[1] for item in entries]
        return ranges, metadata

    def _normalise_overlays(self, payload: dict[str, Any]) -> dict[str, bool]:
        overlays = {"heatmap": False, "tracking": False, "pose": False}
        options = payload.get("options")
        if isinstance(options, dict):
            raw_overlays = options.get("overlays") or options.get("overlay")
        else:
            raw_overlays = None
        if raw_overlays is None:
            task = payload.get("task")
            if isinstance(task, str) and task.lower() == "generate_heatmap":
                overlays["heatmap"] = True
            return overlays
        if isinstance(raw_overlays, dict):
            for name, enabled in raw_overlays.items():
                key = str(name).lower()
                if key in overlays and bool(enabled):
                    overlays[key] = True
            return overlays
        if isinstance(raw_overlays, (list, tuple, set)):
            iterable = raw_overlays
        else:
            iterable = [part.strip() for part in str(raw_overlays).split(",")]
        for item in iterable:
            key = str(item).lower()
            if key in overlays:
                overlays[key] = True
        return overlays

    def _render_overlays(
        self,
        *,
        source_uri: str,
        overlays: dict[str, bool],
        fps_override: float | None,
        artifact_dir: Path,
        clips: Any = None,
    ) -> tuple[Path | None, dict[str, Any], str | None]:
        capture = cv2.VideoCapture(source_uri)
        metadata: dict[str, Any] = {
            "source_uri": source_uri,
            "overlays_requested": [
                name for name, enabled in overlays.items() if enabled
            ],
            "frames": 0,
            "fps": 0.0,
            "codec": None,
            "container": None,
            "path": None,
            "attempts": [],
            "fallback_used": False,
        }
        if not capture.isOpened():
            capture.release()
            metadata["error"] = "unable to open video source"
            return None, metadata, "unable to open video source"

        fps_value = float(fps_override or capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps_value <= 0:
            fps_value = 24.0
        metadata["fps"] = fps_value

        clip_ranges, clip_metadata = self._resolve_clip_ranges(clips, fps_value)
        metadata["clips_requested"] = clip_metadata
        metadata["clips_applied"] = []

        writer = None
        video_path: Path | None = None
        codec_used: str | None = None
        fallback_used = False
        attempts: list[dict[str, Any]] = []
        frame_index = 0
        input_index = 0
        range_index = 0
        applied_range_indices: set[int] = set()
        overlays_rendered: set[str] = set()

        while True:
            success, frame = capture.read()
            if not success:
                break

            if clip_ranges:
                while (
                    range_index < len(clip_ranges)
                    and input_index >= clip_ranges[range_index][1]
                ):
                    range_index += 1
                if range_index >= len(clip_ranges):
                    break
                start_frame, _ = clip_ranges[range_index]
                if input_index < start_frame:
                    input_index += 1
                    continue

            if writer is None:
                height, width = frame.shape[:2]
                (
                    writer,
                    video_path,
                    codec_used,
                    attempts,
                    fallback_used,
                ) = self._initialise_video_writer(
                    artifact_dir=artifact_dir,
                    fps=fps_value,
                    width=width,
                    height=height,
                )
                metadata["attempts"] = attempts
                metadata["fallback_used"] = fallback_used
                if writer is None or video_path is None or codec_used is None:
                    capture.release()
                    metadata["error"] = "unable to initialise video writer"
                    return None, metadata, "unable to initialise video writer"
                metadata["path"] = str(video_path)
                metadata["codec"] = codec_used
                metadata["container"] = video_path.suffix
                metadata["resolution"] = {"width": width, "height": height}

            processed_frame, applied = self._apply_overlays(frame, overlays, frame_index)
            overlays_rendered.update(applied)
            writer.write(processed_frame)
            frame_index += 1
            if clip_ranges:
                applied_range_indices.add(range_index)
            input_index += 1

        capture.release()
        if writer is not None:
            writer.release()

        metadata["frames"] = frame_index
        metadata["overlays_rendered"] = sorted(overlays_rendered)
        if clip_ranges:
            metadata["clips_applied"] = [
                clip_metadata[index] for index in sorted(applied_range_indices)
            ]

        if frame_index == 0:
            if video_path is not None:
                try:
                    video_path.unlink(missing_ok=True)
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            metadata["error"] = "no frames decoded"
            return None, metadata, "no frames decoded"

        return video_path, metadata, None

    def _initialise_video_writer(
        self,
        *,
        artifact_dir: Path,
        fps: float,
        width: int,
        height: int,
    ) -> tuple[
        cv2.VideoWriter | None,
        Path | None,
        str | None,
        list[dict[str, Any]],
        bool,
    ]:
        attempts: list[dict[str, Any]] = []
        fallback_used = False
        combinations = [
            ("mp4v", artifact_dir / "result.mp4"),
            ("avc1", artifact_dir / "result.mp4"),
            ("H264", artifact_dir / "result.mp4"),
            ("X264", artifact_dir / "result.mp4"),
            ("XVID", artifact_dir / "result.mkv"),
            ("MJPG", artifact_dir / "result.mkv"),
        ]

        for codec, path in combinations:
            attempts.append({"codec": codec, "path": str(path)})
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if writer.isOpened():
                fallback_used = path.suffix != ".mp4"
                return writer, path, codec, attempts, fallback_used
            writer.release()
            try:
                path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

        return None, None, None, attempts, fallback_used

    def _apply_overlays(
        self, frame: Any, overlays: dict[str, bool], frame_index: int
    ) -> tuple[Any, list[str]]:
        output = frame.copy()
        applied: list[str] = []

        if overlays.get("heatmap"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            output = cv2.addWeighted(output, 0.4, heatmap, 0.6, 0)
            applied.append("heatmap")

        if overlays.get("tracking"):
            height, width = output.shape[:2]
            width = max(width, 1)
            radius = max(4, min(height, width) // 10)
            center_x = int((frame_index * 7) % width)
            center_y = int(
                height / 2 + (height / 4) * math.sin(frame_index / 5.0)
            )
            center_y = max(radius, min(height - radius, center_y))
            cv2.circle(output, (center_x, center_y), radius, (0, 255, 0), 2)
            applied.append("tracking")

        if overlays.get("pose"):
            height, width = output.shape[:2]
            base_x = int(width * 0.5)
            base_y = int(height * 0.75)
            joints = [
                (base_x, base_y - int(height * 0.3)),
                (base_x, base_y - int(height * 0.15)),
                (base_x - int(width * 0.15), base_y),
                (base_x + int(width * 0.15), base_y),
                (base_x - int(width * 0.2), base_y - int(height * 0.1)),
                (base_x + int(width * 0.2), base_y - int(height * 0.1)),
            ]
            skeleton = [(0, 1), (1, 4), (1, 5), (4, 2), (5, 3), (2, 3)]
            for start_idx, end_idx in skeleton:
                start = joints[start_idx]
                end = joints[end_idx]
                cv2.line(output, start, end, (255, 255, 0), 2)
            cv2.circle(output, joints[0], max(3, min(height, width) // 20), (0, 255, 255), -1)
            applied.append("pose")

        return output, applied
