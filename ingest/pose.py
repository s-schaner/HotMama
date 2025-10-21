from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PoseResult(dict):
    """
    Pose estimation result for a single detection.

    Expected keys:
    - frame_idx: int - Frame index
    - person_id: int - Track ID of the person (if available)
    - bbox: List[float] - Bounding box [x1, y1, x2, y2] where pose was detected
    - keypoints: List[Dict] - List of keypoint dicts with:
        - name: str - Keypoint name (e.g., 'nose', 'left_shoulder')
        - x: float - X coordinate
        - y: float - Y coordinate
        - z: float - Z coordinate (depth, if available)
        - visibility: float - Visibility score (0.0-1.0)
        - confidence: float - Detection confidence (0.0-1.0)
    - pose_landmarks: Optional - Raw pose landmarks from detector
    """
    pass


class PoseEstimator:
    """
    Pose estimation using MediaPipe Pose or YOLO Pose models.

    Supports:
    - MediaPipe Pose (lightweight, accurate)
    - YOLOv8-pose (fast, good for sports)
    - OpenPose (via external integration)
    """

    # MediaPipe landmark names (33 keypoints)
    MEDIAPIPE_LANDMARKS = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]

    def __init__(
        self,
        model_type: str = "mediapipe",
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        static_image_mode: bool = False,
    ):
        """
        Initialize pose estimator.

        Args:
            model_type: Type of model ('mediapipe', 'yolov8-pose', 'openpose')
            model_complexity: Model complexity (0, 1, 2) - higher is more accurate but slower
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            smooth_landmarks: Apply smoothing filter to landmarks
            enable_segmentation: Enable segmentation mask
            static_image_mode: Process each image independently (vs video tracking)
        """
        self.model_type = model_type.lower()
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.static_image_mode = static_image_mode

        self.pose = None
        self._initialized = False

        logger.info(
            f"PoseEstimator initialized: model={model_type}, complexity={model_complexity}, "
            f"min_conf={min_detection_confidence}"
        )

    def _lazy_init(self):
        """Lazy initialization of pose model."""
        if self._initialized:
            return

        try:
            if self.model_type == "mediapipe":
                import mediapipe as mp
                mp_pose = mp.solutions.pose

                self.pose = mp_pose.Pose(
                    static_image_mode=self.static_image_mode,
                    model_complexity=self.model_complexity,
                    smooth_landmarks=self.smooth_landmarks,
                    enable_segmentation=self.enable_segmentation,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.mp_pose = mp_pose  # Keep reference for landmark names
                logger.info("MediaPipe Pose initialized successfully")

            elif self.model_type == "yolov8-pose":
                from ultralytics import YOLO
                self.pose = YOLO("yolov8n-pose.pt")  # Can use yolov8s/m/l/x-pose
                logger.info("YOLOv8-Pose initialized successfully")

            else:
                raise ValueError(
                    f"Unsupported model type: {self.model_type}. "
                    f"Use 'mediapipe' or 'yolov8-pose'"
                )

            self._initialized = True

        except ImportError as e:
            if self.model_type == "mediapipe":
                logger.error("mediapipe not installed. Install with: pip install mediapipe")
                raise ImportError(
                    "MediaPipe pose requires mediapipe. Install with: pip install mediapipe"
                ) from e
            elif self.model_type == "yolov8-pose":
                logger.error("ultralytics not installed. Install with: pip install ultralytics")
                raise ImportError(
                    "YOLO pose requires ultralytics. Install with: pip install ultralytics"
                ) from e
            raise
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}")
            raise

    def infer(
        self,
        frames: Union[Iterable[np.ndarray], np.ndarray],
        bboxes: Optional[List[List[List[float]]]] = None,
        track_ids: Optional[List[List[int]]] = None,
    ) -> List[List[PoseResult]]:
        """
        Run pose estimation on frames.

        Args:
            frames: Iterable of numpy arrays (H, W, C) or single frame
            bboxes: Optional bounding boxes per frame [[bbox1, bbox2, ...], ...]
                   where each bbox is [x1, y1, x2, y2]. If provided, crops to bbox.
            track_ids: Optional track IDs per frame [[id1, id2, ...], ...]

        Returns:
            List of lists of PoseResult dicts, one list per frame
        """
        self._lazy_init()

        # Handle single frame
        if isinstance(frames, np.ndarray):
            if len(frames.shape) == 3:  # Single frame (H, W, C)
                frames = [frames]
                if bboxes is not None:
                    bboxes = [bboxes]
                if track_ids is not None:
                    track_ids = [track_ids]

        results = []

        for frame_idx, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                logger.warning(f"Frame {frame_idx} is not a numpy array, skipping")
                results.append([])
                continue

            frame_bboxes = bboxes[frame_idx] if bboxes else None
            frame_track_ids = track_ids[frame_idx] if track_ids else None

            if self.model_type == "mediapipe":
                frame_results = self._infer_mediapipe(
                    frame, frame_idx, frame_bboxes, frame_track_ids
                )
            elif self.model_type == "yolov8-pose":
                frame_results = self._infer_yolo(
                    frame, frame_idx, frame_bboxes, frame_track_ids
                )
            else:
                frame_results = []

            results.append(frame_results)

        return results

    def _infer_mediapipe(
        self,
        frame: np.ndarray,
        frame_idx: int,
        bboxes: Optional[List[List[float]]] = None,
        track_ids: Optional[List[int]] = None,
    ) -> List[PoseResult]:
        """Run MediaPipe pose estimation on frame."""
        results = []

        # Convert BGR to RGB for MediaPipe
        frame_rgb = frame[..., ::-1].copy() if frame.shape[2] == 3 else frame

        if bboxes:
            # Run pose on each bbox crop
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame_rgb[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                pose_results = self.pose.process(crop)

                if pose_results.pose_landmarks:
                    keypoints = self._parse_mediapipe_landmarks(
                        pose_results.pose_landmarks.landmark,
                        offset_x=x1,
                        offset_y=y1,
                    )

                    results.append(PoseResult(
                        frame_idx=frame_idx,
                        person_id=track_ids[i] if track_ids else i,
                        bbox=bbox,
                        keypoints=keypoints,
                        pose_landmarks=pose_results.pose_landmarks,
                    ))
        else:
            # Run pose on full frame
            pose_results = self.pose.process(frame_rgb)

            if pose_results.pose_landmarks:
                keypoints = self._parse_mediapipe_landmarks(
                    pose_results.pose_landmarks.landmark
                )

                results.append(PoseResult(
                    frame_idx=frame_idx,
                    person_id=0,
                    bbox=[0, 0, frame.shape[1], frame.shape[0]],
                    keypoints=keypoints,
                    pose_landmarks=pose_results.pose_landmarks,
                ))

        return results

    def _infer_yolo(
        self,
        frame: np.ndarray,
        frame_idx: int,
        bboxes: Optional[List[List[float]]] = None,
        track_ids: Optional[List[int]] = None,
    ) -> List[PoseResult]:
        """Run YOLO pose estimation on frame."""
        results = []

        # Run inference
        detections = self.pose.predict(
            frame,
            conf=self.min_detection_confidence,
            verbose=False,
        )

        if len(detections) > 0:
            det = detections[0]

            if det.keypoints is not None:
                # Extract keypoints for each detected person
                for i in range(len(det.keypoints)):
                    kpts = det.keypoints[i].xy.cpu().numpy()[0]  # (17, 2)
                    conf = det.keypoints[i].conf.cpu().numpy()[0]  # (17,)

                    # Get bbox
                    if det.boxes is not None and i < len(det.boxes):
                        bbox = det.boxes.xyxy[i].cpu().numpy().tolist()
                    else:
                        bbox = bboxes[i] if bboxes and i < len(bboxes) else [0, 0, 0, 0]

                    # Parse keypoints (YOLO has 17 COCO keypoints)
                    keypoints = self._parse_yolo_keypoints(kpts, conf)

                    results.append(PoseResult(
                        frame_idx=frame_idx,
                        person_id=track_ids[i] if track_ids and i < len(track_ids) else i,
                        bbox=bbox,
                        keypoints=keypoints,
                    ))

        return results

    def _parse_mediapipe_landmarks(
        self,
        landmarks,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Dict]:
        """Parse MediaPipe landmarks into standardized format."""
        keypoints = []

        for idx, landmark in enumerate(landmarks):
            keypoints.append({
                "name": self.MEDIAPIPE_LANDMARKS[idx] if idx < len(self.MEDIAPIPE_LANDMARKS) else f"point_{idx}",
                "x": landmark.x + offset_x,
                "y": landmark.y + offset_y,
                "z": landmark.z,
                "visibility": landmark.visibility,
                "confidence": landmark.visibility,  # MediaPipe uses visibility as confidence
            })

        return keypoints

    def _parse_yolo_keypoints(self, kpts: np.ndarray, conf: np.ndarray) -> List[Dict]:
        """Parse YOLO keypoints into standardized format."""
        # YOLO COCO keypoint names (17 keypoints)
        YOLO_KEYPOINT_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]

        keypoints = []

        for idx, (xy, c) in enumerate(zip(kpts, conf)):
            keypoints.append({
                "name": YOLO_KEYPOINT_NAMES[idx] if idx < len(YOLO_KEYPOINT_NAMES) else f"point_{idx}",
                "x": float(xy[0]),
                "y": float(xy[1]),
                "z": 0.0,
                "visibility": float(c),
                "confidence": float(c),
            })

        return keypoints

    def close(self):
        """Release resources."""
        if self.pose is not None and hasattr(self.pose, 'close'):
            self.pose.close()
        self._initialized = False
        logger.info("PoseEstimator closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
