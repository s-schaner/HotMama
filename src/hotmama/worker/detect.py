"""Detection engine: person detection + ByteTrack over rally chunks.

The complement to the VLM engine: where the VLM narrates, this one measures.
Any detector that yields ``supervision.Detections`` plugs in; the default is
an ultralytics YOLO loaded lazily (torch never becomes a server or CI
dependency). Tracking runs through the ``trackers`` package's ByteTrack —
supervision's own tracker is deprecated for removal.

Output stays pixel-space and attribution-free (D16): track counts,
visibility, and a coarse center-density grid — the seed of heatmaps. Court
coordinates and team mapping arrive with the calibration phase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger("hotmama.worker.detect")

PERSON_CLASS_ID = 0  # COCO


class DetectError(RuntimeError):
    pass


class FrameDetector(Protocol):
    """Anything that turns one BGR frame into supervision Detections."""

    name: str

    def detect(self, frame: Any) -> Any: ...


class UltralyticsDetector:
    """YOLO via ultralytics, imported only when actually constructed."""

    def __init__(self, model_name: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise DetectError(
                "the detect engine's default detector needs ultralytics — "
                "install hotmama[detect]"
            ) from err
        import supervision as sv

        self._sv = sv
        self._model = YOLO(model_name)
        self._confidence = confidence
        self.name = model_name

    def detect(self, frame: Any) -> Any:
        result = self._model(frame, verbose=False, conf=self._confidence)[0]
        detections = self._sv.Detections.from_ultralytics(result)
        return detections[detections.class_id == PERSON_CLASS_ID]


class DetectEngine:
    """AnalysisEngine: rally chunk → tracked players → density observation."""

    def __init__(
        self,
        detector: FrameDetector,
        *,
        stride: int = 3,
        grid: int = 4,
        min_track_frames: int = 3,
    ) -> None:
        self.name = f"detect:{detector.name}"
        self._detector = detector
        self._stride = max(1, stride)
        self._grid = max(2, grid)
        self._min_track_frames = max(1, min_track_frames)

    def analyze(self, clip_path: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            import cv2
        except ImportError as err:
            raise DetectError("the detect engine needs OpenCV — install hotmama[capture]") from err
        try:
            from trackers import ByteTrackTracker
        except ImportError as err:
            raise DetectError(
                "the detect engine needs the 'trackers' package — install hotmama[detect]"
            ) from err

        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            capture.release()
            raise DetectError(f"could not open clip {clip_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        tracker = ByteTrackTracker(
            frame_rate=max(1.0, fps / self._stride),
            track_activation_threshold=0.4,
            high_conf_det_threshold=0.5,
            minimum_consecutive_frames=2,
        )

        width = height = 0
        frames_processed = 0
        frame_track_ids: list[list[int]] = []
        track_frames: dict[int, int] = {}
        track_hits: dict[int, list[tuple[int, int]]] = {}
        track_confidence: dict[int, list[float]] = {}

        try:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % self._stride:
                    frame_index += 1
                    continue
                frame_index += 1
                frames_processed += 1
                height, width = frame.shape[:2]

                tracked = tracker.update(self._detector.detect(frame))
                tracker_ids = tracked.tracker_id
                if tracker_ids is None or len(tracker_ids) == 0:
                    frame_track_ids.append([])
                    continue
                ids_this_frame: list[int] = []
                confidences = (
                    tracked.confidence
                    if tracked.confidence is not None
                    else [None] * len(tracker_ids)
                )
                for (x1, y1, x2, y2), raw_id, det_conf in zip(
                    tracked.xyxy, tracker_ids, confidences, strict=False
                ):
                    track_id = int(raw_id)
                    if track_id < 0:
                        continue  # -1 = unactivated detection, not a track
                    ids_this_frame.append(track_id)
                    track_frames[track_id] = track_frames.get(track_id, 0) + 1
                    col = min(self._grid - 1, int((x1 + x2) / 2 / width * self._grid))
                    row = min(self._grid - 1, int((y1 + y2) / 2 / height * self._grid))
                    track_hits.setdefault(track_id, []).append((row, col))
                    if det_conf is not None:
                        track_confidence.setdefault(track_id, []).append(float(det_conf))
                frame_track_ids.append(ids_this_frame)
        finally:
            capture.release()

        if frames_processed == 0:
            raise DetectError("clip contains no decodable frames")

        # Only tracks that persisted count anywhere — a one-frame ghost is
        # noise, not a player, and must not inflate any statistic.
        persistent = {
            track_id
            for track_id, count in track_frames.items()
            if count >= self._min_track_frames
        }
        per_frame_counts = [
            sum(1 for track_id in ids if track_id in persistent)
            for ids in frame_track_ids
        ]
        grid_counts = [0] * (self._grid * self._grid)
        confidence_values: list[float] = []
        for track_id in persistent:
            for row, col in track_hits.get(track_id, []):
                grid_counts[row * self._grid + col] += 1
            confidence_values.extend(track_confidence.get(track_id, []))

        mean_confidence = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        )
        observation_confidence = (
            min(0.95, max(0.1, mean_confidence)) if persistent else 0.1
        )

        return [
            {
                "kind": "player_tracks",
                "data": {
                    "engine": self.name,
                    "frames_processed": frames_processed,
                    "stride": self._stride,
                    "resolution": {"width": width, "height": height},
                    "players_max": max(per_frame_counts, default=0),
                    "players_avg": round(
                        sum(per_frame_counts) / len(per_frame_counts), 2
                    )
                    if per_frame_counts
                    else 0.0,
                    "distinct_tracks": len(persistent),
                    "grid": self._grid,
                    "center_grid": grid_counts,
                },
                "confidence": round(observation_confidence, 3),
            }
        ]
