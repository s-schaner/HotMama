"""Clip extraction: a wall-clock window + a segment manifest → one video file.

Two grades of output for two audiences:

- ``reencode=True`` (tag clips, for humans): H.264 + faststart so the PWA's
  ``<video>`` element plays it on any phone.
- ``reencode=False`` (rally chunks, for the Well's CV workers): stream copy —
  near-instant, codec-faithful, keyframe-imprecise by up to a GOP, which the
  ±pad already absorbs.

Windows spanning a segment boundary are extracted piecewise and joined with
the concat demuxer. Windows are clamped to what was actually recorded.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .recorder import RecordingManifest


class ClipError(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as err:  # noqa: BLE001 - any failure means "no ffmpeg"
        raise ClipError(
            "ffmpeg not found — install ffmpeg or the hotmama[capture] extra"
        ) from err


@dataclass(frozen=True)
class _Piece:
    source: Path
    offset: float
    duration: float


def _segment_window(
    manifest: RecordingManifest, index: int, fallback_end: datetime
) -> tuple[datetime, datetime]:
    segment = manifest.segments[index]
    start = segment.start_dt()
    end = segment.end_dt()
    if end is None:
        if segment.frames and manifest.fps > 0:
            end = start + timedelta(seconds=segment.frames / manifest.fps)
        else:
            end = fallback_end
    return start, end


def recorded_range(manifest: RecordingManifest) -> tuple[datetime, datetime] | None:
    if not manifest.segments:
        return None
    first = manifest.segments[0].start_dt()
    last_start, last_end = _segment_window(manifest, len(manifest.segments) - 1, first)
    return first, max(last_start, last_end)


def plan_pieces(
    manifest: RecordingManifest,
    recording_dir: Path,
    start: datetime,
    end: datetime,
) -> list[_Piece]:
    span = recorded_range(manifest)
    if span is None:
        raise ClipError("nothing recorded yet")
    recorded_start, recorded_end = span
    start = max(start, recorded_start)
    end = min(end, recorded_end)
    if end <= start:
        raise ClipError("clip window lies outside the recording")

    pieces: list[_Piece] = []
    for index, segment in enumerate(manifest.segments):
        seg_start, seg_end = _segment_window(manifest, index, recorded_end)
        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)
        if overlap_end <= overlap_start:
            continue
        pieces.append(
            _Piece(
                source=recording_dir / segment.path,
                offset=(overlap_start - seg_start).total_seconds(),
                duration=(overlap_end - overlap_start).total_seconds(),
            )
        )
    if not pieces:
        raise ClipError("clip window matched no recorded segment")
    return pieces


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        tail = result.stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise ClipError(f"ffmpeg failed: {' | '.join(tail)}")


def _extract_piece(exe: str, piece: _Piece, out: Path, *, reencode: bool) -> None:
    command = [
        exe,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{piece.offset:.3f}",
        "-t",
        f"{max(piece.duration, 0.1):.3f}",
        "-i",
        str(piece.source),
    ]
    if reencode:
        command += [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "27",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    else:
        command += ["-c", "copy"]
    command.append(str(out))
    _run(command)


def extract_clip(
    manifest: RecordingManifest,
    recording_dir: Path,
    start: datetime,
    end: datetime,
    out_path: Path,
    *,
    reencode: bool,
) -> None:
    """Extract [start, end) from the recording into ``out_path`` (blocking)."""
    exe = ffmpeg_exe()
    pieces = plan_pieces(manifest, recording_dir, start, end)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(pieces) == 1:
        _extract_piece(exe, pieces[0], out_path, reencode=reencode)
        return

    workdir = out_path.parent / f".{out_path.stem}_parts"
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        part_paths: list[Path] = []
        for number, piece in enumerate(pieces):
            part = workdir / f"part_{number}{out_path.suffix}"
            _extract_piece(exe, piece, part, reencode=reencode)
            part_paths.append(part)
        listing = workdir / "concat.txt"
        listing.write_text("".join(f"file '{path}'\n" for path in part_paths))
        _run(
            [
                exe,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                str(out_path),
            ]
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
