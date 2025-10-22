from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - heavy dependency
    import cv2
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from viz.heatmap import (
    CANVAS_SCALE,
    COURT_LENGTH,
    COURT_WIDTH,
    draw_court_background,
    render_with_strategy,
)


class TrajectoryError(RuntimeError):
    """Raised when the trajectory pipeline cannot complete."""


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    h: float
    confidence: float
    frame_index: int
    raw_x: float
    raw_y: float


@dataclass
class TrajectoryArtifacts:
    method: str
    points: List[TrajectoryPoint]
    csv_path: Path
    heatmaps: Dict[str, Path]
    overlay_video: Path | None
    court_track_video: Path | None
    trail_image: Path | None
    logs: List[str] = field(default_factory=list)


def _require_cv() -> None:
    if cv2 is None or np is None:
        raise TrajectoryError(
            "OpenCV with NumPy is required for trajectory extraction"
        ) from _IMPORT_ERROR


def _px_to_meters(
    x_px: float, y_px: float, frame_w: float, frame_h: float
) -> tuple[float, float]:
    if frame_w <= 0 or frame_h <= 0:
        return 0.0, 0.0
    return (x_px / frame_w) * COURT_WIDTH, (y_px / frame_h) * COURT_LENGTH


class _BaseDetector:
    name: str = "base"

    def setup(
        self, frame_shape: tuple[int, int, int]
    ) -> None:  # pragma: no cover - virtual
        _ = frame_shape

    def detect(
        self, frame, frame_idx: int
    ) -> tuple[int, int, float, float] | None:  # pragma: no cover - virtual
        raise NotImplementedError


class _BackgroundDetector(_BaseDetector):
    name = "background_subtraction"

    def __init__(self) -> None:
        self.subtractor = None

    def setup(self, frame_shape: tuple[int, int, int]) -> None:
        _require_cv()
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=320, varThreshold=36, detectShadows=False
        )

    def detect(self, frame, frame_idx: int) -> tuple[int, int, float, float] | None:
        _ = frame_idx
        if self.subtractor is None:
            self.setup(frame.shape)
        mask = self.subtractor.apply(frame)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), dtype=np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        frame_area = frame.shape[0] * frame.shape[1]
        candidates = [
            c for c in contours if 8 <= cv2.contourArea(c) <= max(frame_area * 0.02, 12)
        ]
        if not candidates:
            return None

        contour = min(candidates, key=cv2.contourArea)
        m = cv2.moments(contour)
        if m.get("m00", 0) <= 0:
            return None
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        area = cv2.contourArea(contour)
        confidence = float(min(1.0, max(0.2, area / max(frame_area * 0.015, 1e-6))))
        height_est = float(math.sqrt(area) * 0.05)
        return cx, cy, height_est, confidence


class _ColorDetector(_BaseDetector):
    name = "color_orange"

    def detect(self, frame, frame_idx: int) -> tuple[int, int, float, float] | None:
        _ = frame_idx
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([5, 80, 70], dtype=np.uint8)
        upper = np.array([28, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 6:
            return None
        m = cv2.moments(contour)
        if m.get("m00", 0) <= 0:
            return None
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        confidence = float(min(1.0, 0.3 + (area / 4000.0)))
        height_est = float(min(3.0, 0.1 + area**0.5 * 0.06))
        return cx, cy, height_est, confidence


class _BrightSpotDetector(_BaseDetector):
    name = "bright_spot"

    def detect(self, frame, frame_idx: int) -> tuple[int, int, float, float] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        min_val, max_val, _, max_loc = cv2.minMaxLoc(blur)
        if max_val - min_val < 8:
            return None
        cx, cy = int(max_loc[0]), int(max_loc[1])
        confidence = float(min(1.0, (max_val - min_val) / 128.0))
        height_est = float(max(0.05, confidence * 0.8))
        return cx, cy, height_est, confidence


_DETECTOR_REGISTRY: Dict[str, type[_BaseDetector]] = {
    cls.name: cls for cls in (_BackgroundDetector, _ColorDetector, _BrightSpotDetector)
}


def list_detection_methods() -> List[str]:
    return list(_DETECTOR_REGISTRY.keys()) + ["auto"]


def _instantiate_detectors(order: Sequence[str]) -> List[_BaseDetector]:
    detectors: List[_BaseDetector] = []
    for name in order:
        cls = _DETECTOR_REGISTRY.get(name)
        if cls is None:
            continue
        detectors.append(cls())
    return detectors


def run_heatmap_pipeline(
    video_path: str | Path,
    output_dir: Path,
    method: str = "auto",
    stride: int = 1,
    smoothing_window: int = 3,
    min_confidence: float = 0.05,
    overlay: bool = True,
    heatmap_strategies: Iterable[str] | None = None,
    renderer_options: Dict[str, dict] | None = None,
) -> TrajectoryArtifacts:
    _require_cv()

    if stride <= 0:
        raise ValueError("stride must be >= 1")

    video_path = Path(video_path)
    if not video_path.exists():
        raise TrajectoryError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise TrajectoryError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0

    method = method.lower().strip() or "auto"
    if method == "auto":
        detector_order = ["background_subtraction", "color_orange", "bright_spot"]
    else:
        detector_order = [method]
    detectors = _instantiate_detectors(detector_order)
    if not detectors:
        raise TrajectoryError(f"Unsupported detection method: {method}")

    for detector in detectors:
        detector.setup((int(frame_h), int(frame_w), 3))

    points: List[TrajectoryPoint] = []
    px_positions: Dict[int, tuple[int, int]] = {}
    logs: List[str] = []

    frame_idx = 0
    success, frame = cap.read()
    while success:
        if frame_idx % stride == 0:
            detection = None
            for detector in detectors:
                detection = detector.detect(frame, frame_idx)
                if detection:
                    break
            if detection:
                cx, cy, h_est, confidence = detection
                if confidence >= min_confidence:
                    t_sec = frame_idx / fps if fps > 0 else float(frame_idx)
                    x_m, y_m = _px_to_meters(cx, cy, frame_w, frame_h)
                    points.append(
                        TrajectoryPoint(
                            t=float(t_sec),
                            x=float(x_m),
                            y=float(y_m),
                            h=float(max(0.0, h_est)),
                            confidence=float(confidence),
                            frame_index=int(frame_idx),
                            raw_x=float(cx),
                            raw_y=float(cy),
                        )
                    )
                    px_positions[frame_idx] = (cx, cy)
            else:
                logs.append(f"No detection on frame {frame_idx}")

        success, frame = cap.read()
        frame_idx += 1

    cap.release()

    if not points:
        raise TrajectoryError("No ball trajectory points detected")

    # Optional smoothing of spatial coordinates (moving average)
    if smoothing_window > 1:
        window = max(1, int(smoothing_window))
        smooth_pts: List[TrajectoryPoint] = []
        for idx, point in enumerate(points):
            start = max(0, idx - window + 1)
            subset = points[start : idx + 1]
            avg_x = sum(p.x for p in subset) / len(subset)
            avg_y = sum(p.y for p in subset) / len(subset)
            avg_h = sum(p.h for p in subset) / len(subset)
            smooth_pts.append(
                TrajectoryPoint(
                    t=point.t,
                    x=avg_x,
                    y=avg_y,
                    h=avg_h,
                    confidence=point.confidence,
                    frame_index=point.frame_index,
                    raw_x=point.raw_x,
                    raw_y=point.raw_y,
                )
            )
        points = smooth_pts

    csv_path = output_dir / "ball_trajectory_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_sec", "x", "y", "h", "confidence", "frame_index"])
        for point in points:
            writer.writerow(
                [
                    f"{point.t:.4f}",
                    f"{point.x:.4f}",
                    f"{point.y:.4f}",
                    f"{point.h:.4f}",
                    f"{point.confidence:.3f}",
                    point.frame_index,
                ]
            )

    # Heatmaps via available strategies
    heatmap_paths: Dict[str, Path] = {}
    strategies = list(heatmap_strategies or ["basic", "opencv", "matplotlib"])
    renderer_options = renderer_options or {}
    for strategy in strategies:
        try:
            opts = renderer_options.get(strategy, {})
        except AttributeError:
            opts = {}
        out_path = output_dir / f"heatmap_{strategy}.png"
        try:
            path = render_with_strategy(
                strategy, [(p.x, p.y, p.h) for p in points], out_path, **opts
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            logs.append(f"Renderer {strategy} failed: {exc}")
        else:
            heatmap_paths[strategy] = path

    if not heatmap_paths:
        logs.append("No heatmaps were successfully rendered")

    overlay_video: Path | None = None
    court_track_video: Path | None = None
    trail_image: Path | None = None

    if overlay:
        overlay_video = _render_overlay_video(video_path, output_dir, fps, px_positions)
        court_track_video = _render_court_video(output_dir, fps, points)
        trail_image = _render_trail_snapshot(output_dir, points)

    return TrajectoryArtifacts(
        method=method,
        points=points,
        csv_path=csv_path,
        heatmaps=heatmap_paths,
        overlay_video=overlay_video,
        court_track_video=court_track_video,
        trail_image=trail_image,
        logs=logs,
    )


def _render_overlay_video(
    video_path: Path,
    output_dir: Path,
    fps: float,
    px_positions: Dict[int, tuple[int, int]],
) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():  # pragma: no cover - runtime guard
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = output_dir / "overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps or 24.0, (frame_w, frame_h))
    if not writer.isOpened():  # pragma: no cover - runtime guard
        cap.release()
        return None

    frame_idx = 0
    last_pos = None
    success, frame = cap.read()
    while success:
        pos = px_positions.get(frame_idx, last_pos)
        if pos:
            cv2.circle(
                frame, pos, 12, (0, 120, 255), thickness=-1, lineType=cv2.LINE_AA
            )
            last_pos = pos
        writer.write(frame)
        success, frame = cap.read()
        frame_idx += 1

    writer.release()
    cap.release()
    return out_path if out_path.exists() else None


def _render_court_video(
    output_dir: Path,
    fps: float,
    points: List[TrajectoryPoint],
) -> Path | None:
    if not points:
        return None

    width = int(COURT_WIDTH * CANVAS_SCALE)
    height = int(COURT_LENGTH * CANVAS_SCALE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = output_dir / "topdown.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps or 24.0, (width, height))
    if not writer.isOpened():  # pragma: no cover - runtime guard
        return None

    base = np.array(draw_court_background(width, height), dtype=np.uint8)
    trail: List[tuple[int, int]] = []
    for point in points:
        frame_img = base.copy()
        cx = int(point.x * CANVAS_SCALE)
        cy = int(point.y * CANVAS_SCALE)
        trail.append((cx, cy))
        if len(trail) >= 2:
            cv2.polylines(
                frame_img,
                [np.array(trail, dtype=np.int32)],
                False,
                (0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        cv2.circle(
            frame_img, (cx, cy), 10, (255, 196, 0), thickness=-1, lineType=cv2.LINE_AA
        )
        writer.write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))

    writer.release()
    return out_path if out_path.exists() else None


def _render_trail_snapshot(
    output_dir: Path, points: List[TrajectoryPoint]
) -> Path | None:
    if not points:
        return None
    width = int(COURT_WIDTH * CANVAS_SCALE)
    height = int(COURT_LENGTH * CANVAS_SCALE)
    image = np.array(draw_court_background(width, height), dtype=np.uint8)
    poly = []
    for point in points:
        cx = int(point.x * CANVAS_SCALE)
        cy = int(point.y * CANVAS_SCALE)
        poly.append((cx, cy))
        cv2.circle(
            image, (cx, cy), 8, (255, 196, 0), thickness=-1, lineType=cv2.LINE_AA
        )
    if len(poly) >= 2:
        cv2.polylines(
            image,
            [np.array(poly, dtype=np.int32)],
            False,
            (0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    out_path = output_dir / "trajectory_snapshot.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return out_path
