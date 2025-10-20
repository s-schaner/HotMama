from __future__ import annotations

from typing import Dict, Iterable, List


class FusionEngine:
    """Combine tracking and pose outputs into normalized events."""

    def fuse(self, tracks: Iterable[dict], poses: Iterable[dict]) -> List[Dict]:
        events: List[Dict] = []
        for track in tracks:
            events.append({"event_type": "track", "payload": track})
        for pose in poses:
            events.append({"event_type": "pose", "payload": pose})
        return events
