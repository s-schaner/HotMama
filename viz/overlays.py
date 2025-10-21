"""
Real-time video overlay rendering for player tracking, pose, and analytics.

Supports:
- Bounding box overlays with track IDs
- Pose skeleton overlays
- Player heatmaps on court
- Trail/trajectory visualization
- Per-player statistics overlay
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2
from collections import defaultdict

logger = logging.getLogger(__name__)


class OverlayConfig:
    """Configuration for video overlays."""

    def __init__(
        self,
        # Bounding box options
        draw_boxes: bool = True,
        box_thickness: int = 2,
        box_color: Tuple[int, int, int] = (0, 255, 0),  # BGR
        draw_track_ids: bool = True,
        id_font_scale: float = 0.6,
        id_thickness: int = 2,

        # Pose options
        draw_pose: bool = True,
        pose_point_radius: int = 4,
        pose_line_thickness: int = 2,
        pose_color: Tuple[int, int, int] = (255, 0, 255),  # BGR magenta

        # Heatmap options
        draw_heatmap: bool = True,
        heatmap_alpha: float = 0.3,
        heatmap_colormap: int = cv2.COLORMAP_JET,

        # Trail options
        draw_trails: bool = True,
        trail_length: int = 30,
        trail_thickness: int = 2,
        trail_fade: bool = True,

        # Stats overlay
        draw_stats: bool = False,
        stats_position: Tuple[int, int] = (10, 30),
        stats_font_scale: float = 0.5,
        stats_thickness: int = 1,

        # Performance
        downsample_pose: bool = False,
        antialiasing: bool = True,
    ):
        """Initialize overlay configuration."""
        self.draw_boxes = draw_boxes
        self.box_thickness = box_thickness
        self.box_color = box_color
        self.draw_track_ids = draw_track_ids
        self.id_font_scale = id_font_scale
        self.id_thickness = id_thickness

        self.draw_pose = draw_pose
        self.pose_point_radius = pose_point_radius
        self.pose_line_thickness = pose_line_thickness
        self.pose_color = pose_color

        self.draw_heatmap = draw_heatmap
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_colormap = heatmap_colormap

        self.draw_trails = draw_trails
        self.trail_length = trail_length
        self.trail_thickness = trail_thickness
        self.trail_fade = trail_fade

        self.draw_stats = draw_stats
        self.stats_position = stats_position
        self.stats_font_scale = stats_font_scale
        self.stats_thickness = stats_thickness

        self.downsample_pose = downsample_pose
        self.antialiasing = antialiasing


class VideoOverlay:
    """
    Render overlays on video frames for player tracking visualization.

    Manages:
    - Bounding boxes with track IDs
    - Pose skeletons
    - Movement trails
    - Heatmaps
    - Statistics overlays
    """

    # Pose skeleton connections (COCO format)
    POSE_CONNECTIONS = [
        # Face
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),

        # Upper body
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),

        # Torso
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),

        # Lower body
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    # Color palette for different track IDs
    TRACK_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark red
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark blue
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]

    def __init__(self, config: Optional[OverlayConfig] = None):
        """Initialize overlay renderer."""
        self.config = config or OverlayConfig()

        # Track history for trails
        self.track_trails = defaultdict(list)  # track_id -> [(x, y), ...]

        # Player stats cache
        self.player_stats = defaultdict(dict)  # track_id -> {stat: value}

        # Heatmap accumulator
        self.heatmap_data = None

        logger.info("VideoOverlay initialized")

    def render_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        poses: Optional[List[Dict]] = None,
        stats: Optional[Dict[int, Dict]] = None,
    ) -> np.ndarray:
        """
        Render overlays on a single frame.

        Args:
            frame: Input frame (H, W, C) BGR
            tracks: List of track dicts with keys: track_id, bbox, score, etc.
            poses: Optional list of pose dicts with keys: person_id, keypoints, etc.
            stats: Optional per-player stats {track_id: {stat: value}}

        Returns:
            Frame with overlays rendered
        """
        output = frame.copy()

        # Update player stats cache
        if stats:
            self.player_stats.update(stats)

        # Draw trails first (below everything)
        if self.config.draw_trails:
            output = self._draw_trails(output, tracks)

        # Draw bounding boxes
        if self.config.draw_boxes:
            output = self._draw_boxes(output, tracks)

        # Draw poses
        if self.config.draw_pose and poses:
            output = self._draw_poses(output, poses)

        # Draw stats overlay
        if self.config.draw_stats:
            output = self._draw_stats_overlay(output, tracks)

        return output

    def _draw_boxes(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Draw bounding boxes with track IDs."""
        for track in tracks:
            track_id = track.get("track_id", 0)
            bbox = track.get("bbox", [])
            score = track.get("score", 0.0)

            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Get color for this track
            color = self._get_track_color(track_id)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                self.config.box_thickness,
            )

            # Draw track ID and confidence
            if self.config.draw_track_ids:
                label = f"ID:{track_id} {score:.2f}"
                label_size, _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.id_font_scale,
                    self.config.id_thickness,
                )

                # Background for text
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 10, y1),
                    color,
                    -1,  # Filled
                )

                # Text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.id_font_scale,
                    (255, 255, 255),  # White text
                    self.config.id_thickness,
                )

        return frame

    def _draw_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw pose skeletons."""
        for pose in poses:
            keypoints = pose.get("keypoints", [])
            person_id = pose.get("person_id", 0)

            # Build keypoint lookup
            kp_dict = {kp["name"]: kp for kp in keypoints}

            # Get color for this person
            color = self._get_track_color(person_id)

            # Draw connections
            for start_name, end_name in self.POSE_CONNECTIONS:
                if start_name in kp_dict and end_name in kp_dict:
                    start_kp = kp_dict[start_name]
                    end_kp = kp_dict[end_name]

                    # Check visibility
                    if start_kp.get("visibility", 0) > 0.5 and end_kp.get("visibility", 0) > 0.5:
                        start_pt = (int(start_kp["x"]), int(start_kp["y"]))
                        end_pt = (int(end_kp["x"]), int(end_kp["y"]))

                        cv2.line(
                            frame,
                            start_pt,
                            end_pt,
                            color,
                            self.config.pose_line_thickness,
                        )

            # Draw keypoints
            for kp in keypoints:
                if kp.get("visibility", 0) > 0.5:
                    pt = (int(kp["x"]), int(kp["y"]))
                    cv2.circle(
                        frame,
                        pt,
                        self.config.pose_point_radius,
                        color,
                        -1,  # Filled
                    )

        return frame

    def _draw_trails(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Draw movement trails for tracked players."""
        # Update trails
        for track in tracks:
            track_id = track.get("track_id", 0)
            bbox = track.get("bbox", [])

            if len(bbox) == 4:
                # Use center of bbox for trail
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)

                self.track_trails[track_id].append((center_x, center_y))

                # Keep only recent positions
                if len(self.track_trails[track_id]) > self.config.trail_length:
                    self.track_trails[track_id].pop(0)

        # Draw trails
        for track_id, trail in self.track_trails.items():
            if len(trail) < 2:
                continue

            color = self._get_track_color(track_id)

            for i in range(len(trail) - 1):
                # Calculate alpha based on position in trail
                if self.config.trail_fade:
                    alpha = (i + 1) / len(trail)
                    thickness = max(1, int(self.config.trail_thickness * alpha))
                else:
                    thickness = self.config.trail_thickness

                cv2.line(
                    frame,
                    trail[i],
                    trail[i + 1],
                    color,
                    thickness,
                )

        return frame

    def _draw_stats_overlay(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Draw statistics overlay."""
        y_offset = self.config.stats_position[1]

        # Title
        cv2.putText(
            frame,
            "Player Stats:",
            self.config.stats_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.stats_font_scale,
            (255, 255, 255),
            self.config.stats_thickness,
        )
        y_offset += 20

        # Draw stats for each tracked player
        for track in tracks:
            track_id = track.get("track_id", 0)
            stats = self.player_stats.get(track_id, {})

            if not stats:
                continue

            # Format stats line
            stats_line = f"ID {track_id}: "
            stats_line += " | ".join([f"{k}:{v}" for k, v in stats.items()])

            cv2.putText(
                frame,
                stats_line,
                (self.config.stats_position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.stats_font_scale,
                self._get_track_color(track_id),
                self.config.stats_thickness,
            )
            y_offset += 20

        return frame

    def render_heatmap_overlay(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        court_homography: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render heatmap overlay on frame.

        Args:
            frame: Input frame
            heatmap: Heatmap array (H, W) or (H, W, 3)
            court_homography: Optional homography matrix to warp heatmap to court perspective

        Returns:
            Frame with heatmap overlay
        """
        # Ensure heatmap is same size as frame
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        else:
            heatmap_resized = heatmap

        # Convert to color if grayscale
        if len(heatmap_resized.shape) == 2:
            heatmap_colored = cv2.applyColorMap(
                heatmap_resized.astype(np.uint8),
                self.config.heatmap_colormap,
            )
        else:
            heatmap_colored = heatmap_resized

        # Apply homography if provided
        if court_homography is not None:
            heatmap_colored = cv2.warpPerspective(
                heatmap_colored,
                court_homography,
                (frame.shape[1], frame.shape[0]),
            )

        # Blend with frame
        output = cv2.addWeighted(
            frame,
            1.0,
            heatmap_colored,
            self.config.heatmap_alpha,
            0,
        )

        return output

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID."""
        return self.TRACK_COLORS[track_id % len(self.TRACK_COLORS)]

    def reset_trails(self):
        """Reset trail history."""
        self.track_trails.clear()

    def update_stats(self, track_id: int, stats: Dict[str, Any]):
        """Update stats for a player."""
        self.player_stats[track_id].update(stats)


class HeatmapGenerator:
    """Generate heatmaps from player movement data."""

    def __init__(
        self,
        frame_size: Tuple[int, int],
        kernel_size: int = 25,
        decay_factor: float = 0.95,
    ):
        """
        Initialize heatmap generator.

        Args:
            frame_size: (width, height) of video frame
            kernel_size: Size of Gaussian kernel for smoothing
            decay_factor: Temporal decay factor (0.0-1.0)
        """
        self.frame_size = frame_size
        self.kernel_size = kernel_size
        self.decay_factor = decay_factor

        # Initialize heatmap
        self.heatmap = np.zeros((frame_size[1], frame_size[0]), dtype=np.float32)

        logger.info(f"HeatmapGenerator initialized: size={frame_size}")

    def update(self, positions: List[Tuple[int, int]], weights: Optional[List[float]] = None):
        """
        Update heatmap with new positions.

        Args:
            positions: List of (x, y) positions
            weights: Optional weights for each position
        """
        # Apply decay
        self.heatmap *= self.decay_factor

        # Add new positions
        for i, (x, y) in enumerate(positions):
            if 0 <= x < self.frame_size[0] and 0 <= y < self.frame_size[1]:
                weight = weights[i] if weights else 1.0
                self.heatmap[y, x] += weight

        # Apply Gaussian blur for smoothing
        self.heatmap = cv2.GaussianBlur(
            self.heatmap,
            (self.kernel_size, self.kernel_size),
            0,
        )

    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Get current heatmap.

        Args:
            normalize: Normalize to 0-255 range

        Returns:
            Heatmap array (H, W)
        """
        if normalize:
            if self.heatmap.max() > 0:
                normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(self.heatmap, dtype=np.uint8)
            return normalized
        else:
            return self.heatmap.copy()

    def reset(self):
        """Reset heatmap."""
        self.heatmap.fill(0)
