"""Source spec parsing and file-source behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from hotmama.capture.sources import (
    CaptureSource,
    SourceOpenError,
    parse_source_spec,
)

from .util import make_dummy_video


def test_spec_parsing() -> None:
    assert parse_source_spec("0").kind == "device"
    assert parse_source_spec(" 2 ").target == "2"
    assert parse_source_spec("rtsp://cam.local/stream").kind == "stream"
    assert parse_source_spec("http://phone:8080/video").kind == "stream"
    assert parse_source_spec("/tmp/game.mp4").kind == "file"
    with pytest.raises(SourceOpenError):
        parse_source_spec("   ")


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceOpenError, match="not found"):
        CaptureSource(parse_source_spec(str(tmp_path / "nope.mp4")))


def test_file_source_reads_all_frames_then_none(tmp_path: Path) -> None:
    video = make_dummy_video(tmp_path / "in.mp4", seconds=0.5, fps=10)
    source = CaptureSource(parse_source_spec(str(video)))
    try:
        assert source.fps == pytest.approx(10.0, abs=0.5)
        frames = 0
        while source.read() is not None:
            frames += 1
        assert frames == 5
    finally:
        source.release()
