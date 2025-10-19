from __future__ import annotations

from typing import Iterable, List, Dict


class Track(dict):
    pass


class PlayerTracker:
    def update(self, detections: Iterable[Dict]) -> List[Track]:
        return [Track(id=i, detection=det) for i, det in enumerate(detections)]
