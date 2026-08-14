"""Segmented recorder: rollover, manifest journaling, clean finish."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")

from hotmama.capture.recorder import load_manifest, open_recorder  # noqa: E402
from hotmama.capture.sources import parse_source_spec  # noqa: E402

from .util import make_dummy_video  # noqa: E402


def test_recorder_segments_and_finishes(tmp_path: Path) -> None:
    video = make_dummy_video(tmp_path / "in.mp4", seconds=2.5, fps=20)
    out_dir = tmp_path / "rec"

    recorder = open_recorder(
        parse_source_spec(str(video)), out_dir, segment_seconds=1.0
    )
    recorder.start()
    recorder.join(timeout=30.0)
    assert not recorder.is_alive(), "recorder did not finish in time"

    status = recorder.status()
    assert status.state == "finished"
    assert status.error is None
    assert status.frames_total == 50
    assert status.segments >= 2

    manifest = load_manifest(out_dir)
    assert manifest is not None
    assert manifest.finished
    assert manifest.fps == pytest.approx(20.0, abs=0.5)
    assert sum(segment.frames for segment in manifest.segments) == 50
    for segment in manifest.segments:
        assert (out_dir / segment.path).exists()
        assert segment.ended_at is not None
        assert segment.end_dt() >= segment.start_dt()  # type: ignore[operator]

    starts = [segment.start_dt() for segment in manifest.segments]
    assert starts == sorted(starts)


def test_recorder_stop_finalizes(tmp_path: Path) -> None:
    video = make_dummy_video(tmp_path / "in.mp4", seconds=4.0, fps=20)
    out_dir = tmp_path / "rec"

    recorder = open_recorder(
        parse_source_spec(str(video)), out_dir, segment_seconds=10.0
    )
    recorder.start()
    import time

    time.sleep(0.8)
    recorder.stop()
    recorder.join(timeout=15.0)

    manifest = load_manifest(out_dir)
    assert manifest is not None
    assert manifest.finished
    assert len(manifest.segments) == 1
    assert manifest.segments[0].ended_at is not None
    assert manifest.segments[0].frames > 0
