"""Synthetic video generation — no fixture binaries in the repo."""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np  # noqa: E402 - ships with opencv


def make_dummy_video(
    path: Path,
    *,
    seconds: float = 3.0,
    fps: float = 20.0,
    size: tuple[int, int] = (160, 120),
) -> Path:
    """Write a small test video with per-frame distinct content."""
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )
        assert writer.isOpened(), "no usable codec for test video"
    total = int(seconds * fps)
    for index in range(total):
        frame = np.full((height, width, 3), (index * 3) % 255, dtype=np.uint8)
        cv2.putText(
            frame,
            str(index),
            (10, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
    writer.release()
    return path


def frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        count = 0
        while True:
            success, _ = capture.read()
            if not success:
                return count
            count += 1
    finally:
        capture.release()
