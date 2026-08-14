"""Pure projections over the event log's derived state."""

from .projections import (
    biggest_leak,
    match_summary,
    player_stats,
    rotation_table,
    scoring_runs,
)

__all__ = [
    "biggest_leak",
    "match_summary",
    "player_stats",
    "rotation_table",
    "scoring_runs",
]
