from __future__ import annotations

from typing import Iterable, List, Dict


class BallTrack(dict):
    pass


class BallTracker:
    def update(self, detections: Iterable[Dict]) -> List[BallTrack]:
        return [BallTrack(id=i, detection=det) for i, det in enumerate(detections)]
