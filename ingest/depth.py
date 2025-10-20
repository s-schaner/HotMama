from __future__ import annotations

from typing import Iterable, List


class DepthMap(dict):
    pass


class DepthEstimator:
    def infer(self, frames: Iterable[int]) -> List[DepthMap]:
        return [DepthMap(frame=frame, depth=0.0) for frame in frames]
