from __future__ import annotations

import logging
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class Track(dict):
    """
    Track object representing a tracked player across frames.

    Expected keys:
    - track_id: int - Unique persistent ID for this track
    - bbox: List[float] - Bounding box [x1, y1, x2, y2]
    - score: float - Detection confidence
    - class_id: int - Object class ID
    - class_name: str - Object class name
    - age: int - Number of frames this track has been alive
    - hits: int - Number of successful detections
    - time_since_update: int - Frames since last detection
    - velocity: Tuple[float, float] - Velocity (dx, dy) in pixels/frame
    """

    pass


class PlayerTracker:
    """
    Multi-object tracker for maintaining persistent player IDs across frames.

    Uses a simplified SORT (Simple Online and Realtime Tracking) algorithm with:
    - IoU-based association
    - Kalman filtering for motion prediction (optional)
    - Track lifecycle management (birth, death, re-identification)
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_kalman: bool = False,
    ):
        """
        Initialize tracker.

        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum detections before track is confirmed
            iou_threshold: Minimum IoU for detection-track association
            use_kalman: Use Kalman filter for motion prediction (requires filterpy)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_kalman = use_kalman

        self.tracks: List[Dict] = []
        self.next_id = 1
        self.frame_count = 0

        # Track history for analytics
        self.track_history = defaultdict(list)  # track_id -> [(frame, bbox), ...]

        logger.info(
            f"PlayerTracker initialized: max_age={max_age}, min_hits={min_hits}, "
            f"iou_threshold={iou_threshold}, use_kalman={use_kalman}"
        )

    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts with keys:
                - bbox or boxes: [x1, y1, x2, y2]
                - score or scores: confidence
                - class_id or classes: class ID
                - class_name or class_names: class name (optional)

        Returns:
            List of Track objects with persistent IDs
        """
        self.frame_count += 1

        # Parse detections into standardized format
        parsed_dets = self._parse_detections(detections)

        # Predict existing track locations
        self._predict_tracks()

        # Associate detections with existing tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(parsed_dets)

        # Update matched tracks
        for det_idx, track_idx in matched:
            self._update_track(track_idx, parsed_dets[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(parsed_dets[det_idx])

        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]["time_since_update"] += 1

        # Remove dead tracks
        self._prune_tracks()

        # Return confirmed tracks
        return self._get_confirmed_tracks()

    def _parse_detections(self, detections: List[Dict]) -> List[Dict]:
        """Parse detections into standardized format."""
        parsed = []

        for det in detections:
            # Handle different detection formats
            bbox = det.get("bbox") or det.get("boxes") or det.get("box")
            score = det.get("score") or det.get("scores") or det.get("confidence", 0.0)
            class_id = det.get("class_id") or det.get("classes") or det.get("class", 0)
            class_name = det.get("class_name") or det.get("class_names") or "person"

            # Handle list formats
            if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], list):
                # Multiple bboxes in one detection - expand
                for i, box in enumerate(bbox):
                    s = score[i] if isinstance(score, list) else score
                    c = class_id[i] if isinstance(class_id, list) else class_id
                    cn = class_name[i] if isinstance(class_name, list) else class_name
                    parsed.append(
                        {
                            "bbox": box,
                            "score": s,
                            "class_id": c,
                            "class_name": cn,
                        }
                    )
            elif bbox:
                parsed.append(
                    {
                        "bbox": bbox,
                        "score": score if not isinstance(score, list) else score[0],
                        "class_id": (
                            class_id if not isinstance(class_id, list) else class_id[0]
                        ),
                        "class_name": (
                            class_name
                            if not isinstance(class_name, list)
                            else class_name[0]
                        ),
                    }
                )

        return parsed

    def _predict_tracks(self):
        """Predict track locations for current frame."""
        for track in self.tracks:
            if self.use_kalman and "kf" in track:
                # Kalman filter prediction
                track["predicted_bbox"] = self._kalman_predict(track["kf"])
            else:
                # Simple constant velocity model
                vx, vy = track.get("velocity", (0, 0))
                x1, y1, x2, y2 = track["bbox"]
                track["predicted_bbox"] = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]

    def _associate(
        self, detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with tracks using IoU.

        Returns:
            matched: [(det_idx, track_idx), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_tracks: [track_idx, ...]
        """
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))

        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                pred_bbox = track.get("predicted_bbox", track["bbox"])
                iou_matrix[d, t] = self._compute_iou(det["bbox"], pred_bbox)

        # Greedy matching (for simplicity; Hungarian algorithm would be better)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Sort by IoU (highest first)
        matches = []
        for d in range(len(detections)):
            for t in range(len(self.tracks)):
                if iou_matrix[d, t] >= self.iou_threshold:
                    matches.append((iou_matrix[d, t], d, t))

        matches.sort(reverse=True, key=lambda x: x[0])

        # Greedy assignment
        used_dets = set()
        used_tracks = set()

        for iou, d, t in matches:
            if d not in used_dets and t not in used_tracks:
                matched.append((d, t))
                used_dets.add(d)
                used_tracks.add(t)

        unmatched_dets = [d for d in range(len(detections)) if d not in used_dets]
        unmatched_tracks = [t for t in range(len(self.tracks)) if t not in used_tracks]

        return matched, unmatched_dets, unmatched_tracks

    def _update_track(self, track_idx: int, detection: Dict):
        """Update existing track with new detection."""
        track = self.tracks[track_idx]

        # Update bbox
        old_bbox = track["bbox"]
        new_bbox = detection["bbox"]
        track["bbox"] = new_bbox

        # Update velocity
        dx = (new_bbox[0] + new_bbox[2]) / 2 - (old_bbox[0] + old_bbox[2]) / 2
        dy = (new_bbox[1] + new_bbox[3]) / 2 - (old_bbox[1] + old_bbox[3]) / 2
        track["velocity"] = (dx, dy)

        # Update other fields
        track["score"] = detection["score"]
        track["class_id"] = detection["class_id"]
        track["class_name"] = detection["class_name"]
        track["hits"] += 1
        track["time_since_update"] = 0
        track["age"] += 1

        # Update history
        self.track_history[track["track_id"]].append((self.frame_count, new_bbox))

        # Update Kalman filter if enabled
        if self.use_kalman and "kf" in track:
            self._kalman_update(track["kf"], new_bbox)

    def _create_track(self, detection: Dict):
        """Create new track from detection."""
        track = {
            "track_id": self.next_id,
            "bbox": detection["bbox"],
            "score": detection["score"],
            "class_id": detection["class_id"],
            "class_name": detection["class_name"],
            "age": 1,
            "hits": 1,
            "time_since_update": 0,
            "velocity": (0.0, 0.0),
        }

        # Initialize Kalman filter if enabled
        if self.use_kalman:
            track["kf"] = self._init_kalman(detection["bbox"])

        self.tracks.append(track)
        self.track_history[self.next_id].append((self.frame_count, detection["bbox"]))
        self.next_id += 1

    def _prune_tracks(self):
        """Remove tracks that have been lost for too long."""
        self.tracks = [t for t in self.tracks if t["time_since_update"] < self.max_age]

    def _get_confirmed_tracks(self) -> List[Track]:
        """Get tracks that have enough hits to be confirmed."""
        confirmed = []

        for track in self.tracks:
            # Track must have minimum hits and recent update
            if track["hits"] >= self.min_hits or track["age"] < self.min_hits:
                confirmed.append(
                    Track(
                        track_id=track["track_id"],
                        bbox=track["bbox"],
                        score=track["score"],
                        class_id=track["class_id"],
                        class_name=track["class_name"],
                        age=track["age"],
                        hits=track["hits"],
                        time_since_update=track["time_since_update"],
                        velocity=track["velocity"],
                    )
                )

        return confirmed

    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bboxes in [x1, y1, x2, y2] format."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _init_kalman(self, bbox: List[float]):
        """Initialize Kalman filter for track (requires filterpy)."""
        try:
            from filterpy.kalman import KalmanFilter

            kf = KalmanFilter(dim_x=7, dim_z=4)
            # [x, y, s, r, vx, vy, vs]
            # x, y: center, s: scale (area), r: aspect ratio
            # vx, vy, vs: velocities
            # TODO: Implement proper Kalman filter setup
            return kf
        except ImportError:
            logger.warning("filterpy not installed, Kalman filtering disabled")
            return None

    def _kalman_predict(self, kf) -> List[float]:
        """Predict next state with Kalman filter."""
        # TODO: Implement Kalman prediction
        return [0, 0, 0, 0]

    def _kalman_update(self, kf, bbox: List[float]):
        """Update Kalman filter with detection."""
        # TODO: Implement Kalman update
        pass

    def get_track_history(self, track_id: int) -> List[Tuple[int, List[float]]]:
        """Get historical positions for a track."""
        return self.track_history.get(track_id, [])

    def get_all_track_histories(self) -> Dict[int, List[Tuple[int, List[float]]]]:
        """Get all track histories."""
        return dict(self.track_history)

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.track_history.clear()
        logger.info("Tracker reset")
