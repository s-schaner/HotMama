from __future__ import annotations

from typing import Dict, Iterable, List


class PoseResult(dict):
    pass


class PoseEstimator:
    def infer(self, frames: Iterable[int]) -> List[PoseResult]:
        return [PoseResult(frame=frame, keypoints=[]) for frame in frames]
