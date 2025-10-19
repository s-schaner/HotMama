from __future__ import annotations

from typing import Dict


class Homography:
    def __init__(self, matrix: Dict[str, float]):
        self.matrix = matrix


class CourtCalibrator:
    def estimate(self, detections) -> Homography:
        return Homography({"h00": 1.0, "h11": 1.0})
