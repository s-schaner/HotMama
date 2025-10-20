from __future__ import annotations

from scoring.fusion import FusionEngine


def test_fusion_engine():
    engine = FusionEngine()
    tracks = [{"id": 1}, {"id": 2}]
    poses = [{"frame": 0}, {"frame": 1}]
    events = engine.fuse(tracks, poses)
    assert len(events) == 4
    assert any(event["event_type"] == "pose" for event in events)
