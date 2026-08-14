"""Clip window math and ffmpeg extraction against synthesized segments."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from hotmama.capture.clips import (  # noqa: E402
    ClipError,
    extract_clip,
    plan_pieces,
    recorded_range,
)
from hotmama.capture.recorder import RecordingManifest  # noqa: E402

from .util import frame_count, make_dummy_video  # noqa: E402

T0 = datetime(2026, 8, 13, 18, 0, 0, tzinfo=UTC)
FPS = 20.0


@pytest.fixture()
def recording(tmp_path: Path) -> tuple[RecordingManifest, Path]:
    """Two 2-second segments with known wall-clock placement."""
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    seg1 = make_dummy_video(rec_dir / "seg_0001.mp4", seconds=2.0, fps=FPS)
    seg2 = make_dummy_video(rec_dir / "seg_0002.mp4", seconds=2.0, fps=FPS)
    manifest = RecordingManifest.from_dict(
        {
            "codec": "mp4v",
            "container": ".mp4",
            "fps": FPS,
            "width": 160,
            "height": 120,
            "finished": True,
            "segments": [
                {
                    "index": 1,
                    "path": seg1.name,
                    "started_at": T0.isoformat(),
                    "ended_at": (T0 + timedelta(seconds=2)).isoformat(),
                    "frames": 40,
                },
                {
                    "index": 2,
                    "path": seg2.name,
                    "started_at": (T0 + timedelta(seconds=2)).isoformat(),
                    "ended_at": (T0 + timedelta(seconds=4)).isoformat(),
                    "frames": 40,
                },
            ],
        }
    )
    return manifest, rec_dir


def test_recorded_range(recording: tuple[RecordingManifest, Path]) -> None:
    manifest, _ = recording
    span = recorded_range(manifest)
    assert span == (T0, T0 + timedelta(seconds=4))


def test_plan_single_segment(recording: tuple[RecordingManifest, Path]) -> None:
    manifest, rec_dir = recording
    pieces = plan_pieces(
        manifest, rec_dir, T0 + timedelta(seconds=0.5), T0 + timedelta(seconds=1.5)
    )
    assert len(pieces) == 1
    assert pieces[0].offset == pytest.approx(0.5)
    assert pieces[0].duration == pytest.approx(1.0)


def test_plan_clamps_to_recording(recording: tuple[RecordingManifest, Path]) -> None:
    manifest, rec_dir = recording
    pieces = plan_pieces(
        manifest, rec_dir, T0 - timedelta(seconds=30), T0 + timedelta(seconds=1)
    )
    assert len(pieces) == 1
    assert pieces[0].offset == pytest.approx(0.0)
    assert pieces[0].duration == pytest.approx(1.0)


def test_plan_rejects_outside_window(recording: tuple[RecordingManifest, Path]) -> None:
    manifest, rec_dir = recording
    with pytest.raises(ClipError, match="outside"):
        plan_pieces(
            manifest, rec_dir, T0 + timedelta(seconds=10), T0 + timedelta(seconds=11)
        )


def test_plan_crossing_segments(recording: tuple[RecordingManifest, Path]) -> None:
    manifest, rec_dir = recording
    pieces = plan_pieces(
        manifest, rec_dir, T0 + timedelta(seconds=1.5), T0 + timedelta(seconds=2.5)
    )
    assert len(pieces) == 2
    assert pieces[0].offset == pytest.approx(1.5)
    assert pieces[0].duration == pytest.approx(0.5)
    assert pieces[1].offset == pytest.approx(0.0)
    assert pieces[1].duration == pytest.approx(0.5)


def test_extract_reencoded_clip(recording: tuple[RecordingManifest, Path], tmp_path: Path) -> None:
    manifest, rec_dir = recording
    out = tmp_path / "clip.mp4"
    extract_clip(
        manifest,
        rec_dir,
        T0 + timedelta(seconds=0.5),
        T0 + timedelta(seconds=1.5),
        out,
        reencode=True,
    )
    assert out.exists() and out.stat().st_size > 0
    frames = frame_count(out)
    assert frames == pytest.approx(FPS, abs=6)  # ~1s of video


def test_extract_crossing_copy_clip(
    recording: tuple[RecordingManifest, Path], tmp_path: Path
) -> None:
    manifest, rec_dir = recording
    out = tmp_path / "rally.mp4"
    extract_clip(
        manifest,
        rec_dir,
        T0 + timedelta(seconds=1.5),
        T0 + timedelta(seconds=2.5),
        out,
        reencode=False,
    )
    assert out.exists() and out.stat().st_size > 0
    assert frame_count(out) > 0
