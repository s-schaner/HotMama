"""Detect engine: real ByteTrack, fake detector — no torch required."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cv2")
sv = pytest.importorskip("supervision")
pytest.importorskip("trackers")

import numpy as np  # noqa: E402

from hotmama.worker.detect import DetectEngine, DetectError  # noqa: E402
from hotmama.worker.engine import ENGINES, make_engine  # noqa: E402

from ..capture.util import make_dummy_video  # noqa: E402


class FakeDetector:
    """Two steady 'players' drifting right; optional one-frame ghost."""

    name = "fake"

    def __init__(self, ghost_on_frame: int | None = None) -> None:
        self.calls = 0
        self._ghost_on_frame = ghost_on_frame

    def detect(self, frame: Any) -> Any:
        index = self.calls
        self.calls += 1
        drift = index * 2.0
        boxes = [
            [10 + drift, 20, 40 + drift, 100],
            [90 + drift, 25, 120 + drift, 105],
        ]
        confidences = [0.85, 0.8]
        if index == self._ghost_on_frame:
            boxes.append([70, 5, 80, 30])
            confidences.append(0.9)
        return sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.array(confidences, dtype=float),
            class_id=np.zeros(len(boxes), dtype=int),
        )


class EmptyDetector:
    name = "empty"

    def detect(self, frame: Any) -> Any:
        return sv.Detections.empty()


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    # 30 frames at 160x120 — processed at stride 1 for determinism.
    return make_dummy_video(tmp_path / "rally.mp4", seconds=1.5, fps=20, size=(160, 120))


def test_two_players_tracked(clip: Path) -> None:
    engine = DetectEngine(FakeDetector(), stride=1, min_track_frames=3)
    observations = engine.analyze(clip, {"clip_id": "c_1"})

    assert len(observations) == 1
    observation = observations[0]
    assert observation["kind"] == "player_tracks"
    data = observation["data"]
    assert data["frames_processed"] == 30
    assert data["players_max"] == 2
    assert data["players_avg"] >= 1.8  # tracker warmup may miss the first frame
    assert data["distinct_tracks"] == 2
    assert data["resolution"] == {"width": 160, "height": 120}
    assert len(data["center_grid"]) == 16
    hits = sum(data["center_grid"])
    assert 0 < hits <= data["frames_processed"] * 2
    assert 0.5 <= observation["confidence"] <= 0.95


def test_one_frame_ghost_is_not_a_track(clip: Path) -> None:
    engine = DetectEngine(FakeDetector(ghost_on_frame=10), stride=1, min_track_frames=3)
    data = engine.analyze(clip, {})[0]["data"]
    assert data["distinct_tracks"] == 2
    assert data["players_max"] == 2  # single-frame ghost never activates a track


def test_empty_clip_of_people(clip: Path) -> None:
    engine = DetectEngine(EmptyDetector(), stride=1)
    observation = engine.analyze(clip, {})[0]
    assert observation["data"]["distinct_tracks"] == 0
    assert observation["data"]["players_max"] == 0
    assert observation["confidence"] == 0.1


def test_stride_reduces_frames(clip: Path) -> None:
    detector = FakeDetector()
    engine = DetectEngine(detector, stride=3)
    data = engine.analyze(clip, {})[0]["data"]
    assert data["frames_processed"] == 10
    assert detector.calls == 10


def test_unreadable_clip_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(DetectError):
        DetectEngine(FakeDetector(), stride=1).analyze(bad, {})


def test_factory_registered_and_injectable() -> None:
    assert "detect" in ENGINES
    engine = make_engine("detect", {"detector": FakeDetector(), "detect_stride": 2})
    assert engine.name == "detect:fake"


def test_calibrated_job_adds_court_stats(clip: Path) -> None:
    # Full frame == full court; image bottom = near baseline. FakeDetector
    # feet sit at y≈100-105 of 120 → shallow NEAR-half court positions.
    calibration = {
        "image_corners": [[0.0, 120.0], [160.0, 120.0], [160.0, 0.0], [0.0, 0.0]],
        "frame_width": 160,
        "frame_height": 120,
        "mode": "full",
    }
    engine = DetectEngine(FakeDetector(), stride=1, min_track_frames=3)
    data = engine.analyze(clip, {"calibration": calibration})[0]["data"]

    court = data["court"]
    assert court["mode"] == "full"
    assert court["near_hits"] > 0
    assert court["far_hits"] == 0
    assert court["cell_grid_cols"] == 3 and court["cell_grid_rows"] == 6
    assert sum(court["cell_grid"]) == court["near_hits"] + court["far_hits"]
    # Shallow depth: all hits land in the first two 3m bands.
    deep_rows = sum(court["cell_grid"][2 * 3 :])
    assert deep_rows == 0


def test_invalid_calibration_ignored(clip: Path) -> None:
    engine = DetectEngine(FakeDetector(), stride=1)
    data = engine.analyze(clip, {"calibration": {"image_corners": "garbage"}})[0]["data"]
    assert "court" not in data
