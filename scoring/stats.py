from __future__ import annotations

from collections import Counter
from typing import Iterable, List


class StatsAccumulator:
    def __init__(self) -> None:
        self.counter = Counter()

    def ingest(self, events: Iterable[dict]) -> None:
        for event in events:
            key = (
                event.get("actor_team", ""),
                event.get("actor_number", ""),
                event.get("event_type", "unknown"),
            )
            self.counter[key] += 1

    def snapshot(self) -> List[dict]:
        return [
            {
                "actor_team": team,
                "actor_number": number,
                "event_type": event_type,
                "count": count,
            }
            for (team, number, event_type), count in self.counter.items()
        ]
