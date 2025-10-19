from __future__ import annotations

from typing import Iterable, List, Dict


class DetectionResult(dict):
    pass


class DetectorRunner:
    def __init__(self, model: str = "yolov8n") -> None:
        self.model = model

    def detect(self, frames: Iterable[int]) -> List[DetectionResult]:
        return [DetectionResult(frame=frame, boxes=[], scores=[]) for frame in frames]
