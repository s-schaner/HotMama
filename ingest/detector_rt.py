from __future__ import annotations

import logging
from typing import Iterable, List, Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class DetectionResult(dict):
    """
    Detection result for a single frame.

    Expected keys:
    - frame_idx: int - Frame index
    - boxes: List[List[float]] - Bounding boxes in [x1, y1, x2, y2] format
    - scores: List[float] - Confidence scores for each detection
    - classes: List[int] - Class IDs for each detection
    - class_names: List[str] - Human-readable class names
    """
    pass


class DetectorRunner:
    """
    Real-time object detector using YOLO models.

    Supports YOLOv8, YOLOv11, and compatible models via ultralytics library.
    Optimized for volleyball player detection with configurable confidence thresholds
    and GPU acceleration.
    """

    def __init__(
        self,
        model: str = "yolov8n",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        classes_filter: Optional[List[int]] = None,
        enable_tracking: bool = False,
    ) -> None:
        """
        Initialize YOLO detector.

        Args:
            model: Model identifier (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov11n, etc.)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.). Auto-detected if None.
            classes_filter: List of class IDs to detect (None = all, [0] = person only)
            enable_tracking: Enable built-in YOLO tracking
        """
        self.model_name = model
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes_filter = classes_filter or [0]  # Default to person class only
        self.enable_tracking = enable_tracking
        self.model = None
        self._initialized = False

        logger.info(
            f"DetectorRunner initialized: model={model}, conf={confidence_threshold}, "
            f"device={device or 'auto'}, classes={classes_filter}"
        )

    def _lazy_init(self):
        """Lazy initialization of YOLO model to avoid loading during import."""
        if self._initialized:
            return

        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)

            # Move to specified device
            if self.device:
                self.model.to(self.device)

            self._initialized = True
            logger.info(f"YOLO model loaded successfully on device: {self.model.device}")

        except ImportError as e:
            logger.error(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
            raise ImportError(
                "YOLO detection requires ultralytics. Install with: pip install ultralytics"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def detect(
        self,
        frames: Union[Iterable[np.ndarray], np.ndarray],
        verbose: bool = False,
    ) -> List[DetectionResult]:
        """
        Run detection on frames.

        Args:
            frames: Iterable of numpy arrays (H, W, C) or single frame
            verbose: Show detection progress

        Returns:
            List of DetectionResult dicts with boxes, scores, classes, and class names
        """
        self._lazy_init()

        # Handle single frame
        if isinstance(frames, np.ndarray):
            if len(frames.shape) == 3:  # Single frame (H, W, C)
                frames = [frames]

        results = []

        for frame_idx, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                logger.warning(f"Frame {frame_idx} is not a numpy array, skipping")
                continue

            # Run inference
            detections = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes_filter,
                verbose=verbose,
                device=self.device,
            )

            # Extract results
            if len(detections) > 0:
                det = detections[0]  # Single image
                boxes = det.boxes.xyxy.cpu().numpy().tolist() if det.boxes is not None else []
                scores = det.boxes.conf.cpu().numpy().tolist() if det.boxes is not None else []
                classes = det.boxes.cls.cpu().numpy().astype(int).tolist() if det.boxes is not None else []

                # Get class names
                class_names = [det.names[cls_id] for cls_id in classes] if det.names else []

                results.append(
                    DetectionResult(
                        frame_idx=frame_idx,
                        boxes=boxes,
                        scores=scores,
                        classes=classes,
                        class_names=class_names,
                        frame_shape=frame.shape,
                    )
                )
            else:
                # No detections
                results.append(
                    DetectionResult(
                        frame_idx=frame_idx,
                        boxes=[],
                        scores=[],
                        classes=[],
                        class_names=[],
                        frame_shape=frame.shape,
                    )
                )

        logger.debug(f"Detected objects in {len(results)} frames")
        return results

    def detect_video(
        self,
        video_path: str,
        stream: bool = True,
        verbose: bool = False,
    ) -> List[DetectionResult]:
        """
        Run detection on entire video file.

        Args:
            video_path: Path to video file
            stream: Use streaming mode for memory efficiency
            verbose: Show detection progress

        Returns:
            List of DetectionResult dicts for each frame
        """
        self._lazy_init()

        logger.info(f"Running detection on video: {video_path}")

        results_list = []
        frame_idx = 0

        # Run inference on video
        detections = self.model.predict(
            video_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes_filter,
            stream=stream,
            verbose=verbose,
            device=self.device,
        )

        for det in detections:
            boxes = det.boxes.xyxy.cpu().numpy().tolist() if det.boxes is not None else []
            scores = det.boxes.conf.cpu().numpy().tolist() if det.boxes is not None else []
            classes = det.boxes.cls.cpu().numpy().astype(int).tolist() if det.boxes is not None else []
            class_names = [det.names[cls_id] for cls_id in classes] if det.names else []

            results_list.append(
                DetectionResult(
                    frame_idx=frame_idx,
                    boxes=boxes,
                    scores=scores,
                    classes=classes,
                    class_names=class_names,
                )
            )
            frame_idx += 1

        logger.info(f"Completed detection on {frame_idx} frames")
        return results_list

    def get_model_info(self) -> Dict:
        """Get model information."""
        self._lazy_init()
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if self.model else None,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "classes_filter": self.classes_filter,
        }
