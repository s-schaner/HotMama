from __future__ import annotations


def is_point_winner(event: dict) -> bool:
    return event.get("event_type") in {"serve_ace", "ball_down_in", "ball_down_out"}
