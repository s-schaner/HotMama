"""
Integrated video processing pipeline for real-time player tracking and analysis.

Combines:
- YOLO detection
- Multi-object tracking
- Pose estimation
- Video overlays
- Analytics extraction
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass, field

from ingest.detector_rt import DetectorRunner, DetectionResult
from ingest.tracker import PlayerTracker, Track
from ingest.pose import PoseEstimator, PoseResult
from viz.overlays import VideoOverlay, OverlayConfig, HeatmapGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for video processing pipeline."""

    # Detection settings
    yolo_model: str = "yolov8n"
    detection_confidence: float = 0.25
    detection_iou: float = 0.45
    detection_device: Optional[str] = None
    detect_classes: Optional[List[int]] = None

    # Tracking settings
    tracking_max_age: int = 30
    tracking_min_hits: int = 3
    tracking_iou_threshold: float = 0.3
    use_kalman: bool = False

    # Pose settings
    enable_pose: bool = True
    pose_model_type: str = "mediapipe"
    pose_model_complexity: int = 1
    pose_min_confidence: float = 0.5

    # Overlay settings
    enable_overlays: bool = True
    overlay_config: Optional[OverlayConfig] = None

    # Processing settings
    process_every_n_frames: int = 1
    enable_heatmap: bool = True
    output_fps: Optional[int] = None

    # Callbacks
    frame_callback: Optional[Callable[[int, np.ndarray, Dict], None]] = None
    event_callback: Optional[Callable[[Dict], None]] = None


@dataclass
class PipelineResult:
    """Results from video pipeline processing."""

    video_path: str
    total_frames: int
    processed_frames: int
    output_path: Optional[str] = None

    # Tracking data
    tracks_per_frame: List[List[Track]] = field(default_factory=list)
    track_histories: Dict[int, List] = field(default_factory=dict)

    # Pose data
    poses_per_frame: List[List[PoseResult]] = field(default_factory=list)

    # Analytics
    events: List[Dict] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None

    # Metadata
    fps: float = 0.0
    duration: float = 0.0
    frame_size: tuple = (0, 0)


class VideoProcessingPipeline:
    """
    End-to-end video processing pipeline for volleyball player tracking.

    Features:
    - Real-time player detection with YOLO
    - Persistent player tracking across frames
    - Pose estimation for each player
    - Video overlay rendering
    - Movement heatmap generation
    - Event detection and analytics
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.detector = DetectorRunner(
            model=self.config.yolo_model,
            confidence_threshold=self.config.detection_confidence,
            iou_threshold=self.config.detection_iou,
            device=self.config.detection_device,
            classes_filter=self.config.detect_classes or [0],  # Person class
        )

        self.tracker = PlayerTracker(
            max_age=self.config.tracking_max_age,
            min_hits=self.config.tracking_min_hits,
            iou_threshold=self.config.tracking_iou_threshold,
            use_kalman=self.config.use_kalman,
        )

        self.pose_estimator = None
        if self.config.enable_pose:
            self.pose_estimator = PoseEstimator(
                model_type=self.config.pose_model_type,
                model_complexity=self.config.pose_model_complexity,
                min_detection_confidence=self.config.pose_min_confidence,
            )

        self.overlay_renderer = None
        if self.config.enable_overlays:
            overlay_config = self.config.overlay_config or OverlayConfig()
            self.overlay_renderer = VideoOverlay(config=overlay_config)

        self.heatmap_generator = None

        logger.info("VideoProcessingPipeline initialized")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineResult:
        """
        Process entire video with tracking and overlays.

        Args:
            video_path: Path to input video
            output_path: Optional path for output video with overlays
            progress_callback: Optional callback for progress updates (frame_idx, total_frames)

        Returns:
            PipelineResult with tracking data and analytics
        """
        logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Video properties: {total_frames} frames, {fps:.2f} fps, "
            f"{width}x{height}, {duration:.2f}s"
        )

        # Initialize heatmap generator
        if self.config.enable_heatmap:
            self.heatmap_generator = HeatmapGenerator(
                frame_size=(width, height),
                kernel_size=25,
                decay_factor=0.95,
            )

        # Initialize video writer if output requested
        writer = None
        if output_path:
            output_fps = self.config.output_fps or fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            logger.info(f"Writing output to: {output_path}")

        # Initialize result
        result = PipelineResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=0,
            output_path=output_path,
            fps=fps,
            duration=duration,
            frame_size=(width, height),
        )

        # Process frames
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if configured
                if frame_idx % self.config.process_every_n_frames != 0:
                    if writer:
                        writer.write(frame)
                    frame_idx += 1
                    continue

                # Process frame
                frame_data = self._process_frame(frame, frame_idx)

                # Store results
                result.tracks_per_frame.append(frame_data["tracks"])
                if self.config.enable_pose:
                    result.poses_per_frame.append(frame_data["poses"])

                # Update heatmap
                if self.heatmap_generator and frame_data["tracks"]:
                    positions = []
                    for track in frame_data["tracks"]:
                        bbox = track.get("bbox", [])
                        if len(bbox) == 4:
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)
                            positions.append((center_x, center_y))
                    self.heatmap_generator.update(positions)

                # Render overlays
                output_frame = frame
                if self.overlay_renderer:
                    output_frame = self.overlay_renderer.render_frame(
                        frame,
                        frame_data["tracks"],
                        frame_data.get("poses"),
                    )

                # Write output frame
                if writer:
                    writer.write(output_frame)

                # Frame callback
                if self.config.frame_callback:
                    self.config.frame_callback(frame_idx, output_frame, frame_data)

                # Progress callback
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                frame_idx += 1
                result.processed_frames = frame_idx

        finally:
            cap.release()
            if writer:
                writer.release()

        # Get final heatmap
        if self.heatmap_generator:
            result.heatmap = self.heatmap_generator.get_heatmap(normalize=True)

        # Get track histories
        result.track_histories = self.tracker.get_all_track_histories()

        logger.info(f"Processed {result.processed_frames} frames")
        return result

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input frame
            frame_idx: Frame index

        Returns:
            Dict with detection, tracking, and pose results
        """
        frame_data = {
            "frame_idx": frame_idx,
            "detections": [],
            "tracks": [],
            "poses": [],
        }

        # 1. Detect players
        detections = self.detector.detect([frame], verbose=False)
        if detections:
            frame_data["detections"] = detections[0]

        # 2. Update tracker
        tracks = self.tracker.update([frame_data["detections"]])
        frame_data["tracks"] = tracks

        # 3. Pose estimation (per tracked player)
        if self.pose_estimator and tracks:
            # Extract bounding boxes and track IDs
            bboxes = [track.get("bbox") for track in tracks]
            track_ids = [track.get("track_id") for track in tracks]

            # Run pose estimation
            poses = self.pose_estimator.infer(
                [frame],
                bboxes=[bboxes],
                track_ids=[track_ids],
            )

            if poses:
                frame_data["poses"] = poses[0]

        return frame_data

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame (stateful - maintains tracking).

        Args:
            frame: Input frame

        Returns:
            Dict with tracking and pose data
        """
        return self._process_frame(frame, 0)

    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        if self.overlay_renderer:
            self.overlay_renderer.reset_trails()
        if self.heatmap_generator:
            self.heatmap_generator.reset()
        logger.info("Pipeline reset")

    def extract_events(self, result: PipelineResult) -> List[Dict]:
        """
        Extract volleyball events from tracking and pose data.

        This is a stub that can be extended with ML-based event detection.

        Args:
            result: Pipeline result with tracking data

        Returns:
            List of detected events
        """
        events = []

        # TODO: Implement event detection logic
        # - Jump detection (from pose)
        # - Attack/spike detection
        # - Serve detection
        # - Ball contact detection
        # etc.

        return events

    def get_player_analytics(self, result: PipelineResult) -> Dict[int, Dict]:
        """
        Compute per-player analytics from tracking data.

        Args:
            result: Pipeline result

        Returns:
            Dict mapping track_id to analytics dict
        """
        analytics = {}

        for track_id, history in result.track_histories.items():
            if not history:
                continue

            # Extract positions
            positions = [bbox for frame, bbox in history]

            # Compute basic metrics
            analytics[track_id] = {
                "total_frames": len(positions),
                "avg_x": np.mean([((b[0] + b[2]) / 2) for b in positions]),
                "avg_y": np.mean([((b[1] + b[3]) / 2) for b in positions]),
                "movement_distance": self._compute_movement_distance(positions),
                "avg_speed": self._compute_avg_speed(positions, result.fps),
            }

        return analytics

    @staticmethod
    def _compute_movement_distance(bboxes: List[List[float]]) -> float:
        """Compute total movement distance from bboxes."""
        distance = 0.0

        for i in range(len(bboxes) - 1):
            x1 = (bboxes[i][0] + bboxes[i][2]) / 2
            y1 = (bboxes[i][1] + bboxes[i][3]) / 2
            x2 = (bboxes[i + 1][0] + bboxes[i + 1][2]) / 2
            y2 = (bboxes[i + 1][1] + bboxes[i + 1][3]) / 2

            distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return distance

    @staticmethod
    def _compute_avg_speed(bboxes: List[List[float]], fps: float) -> float:
        """Compute average speed in pixels per second."""
        if len(bboxes) < 2 or fps == 0:
            return 0.0

        total_distance = VideoProcessingPipeline._compute_movement_distance(bboxes)
        total_time = len(bboxes) / fps

        return total_distance / total_time if total_time > 0 else 0.0
