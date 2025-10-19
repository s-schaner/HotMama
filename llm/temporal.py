from __future__ import annotations

from typing import Iterable, List


class TemporalEncoder:
    def encode(self, frames: Iterable[int]) -> List[float]:
        return [float(frame) for frame in frames]
