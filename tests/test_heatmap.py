from __future__ import annotations

from pathlib import Path

import pytest

from viz.heatmap import render_heatmap, render_with_strategy
from viz.trajectory import run_heatmap_pipeline

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional deps
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


def test_render_heatmap(tmp_path: Path) -> None:
    points = [(1.0, 1.0, 0.1), (2.0, 5.0, 0.9)]
    out_path = tmp_path / "heatmap.png"
    result = render_heatmap(points, out_path)
    assert result.exists()
    assert result.stat().st_size > 0


def test_render_with_strategy_basic(tmp_path: Path) -> None:
    points = [(1.0, 1.0, 0.1), (4.5, 7.0, 1.2)]
    out_path = tmp_path / "basic.png"
    render_with_strategy("basic", points, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.mark.skipif(cv2 is None or np is None, reason="OpenCV + NumPy required")
def test_run_heatmap_pipeline(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _make_dummy_video(video_path)
    out_dir = tmp_path / "out"
    artifacts = run_heatmap_pipeline(
        video_path,
        out_dir,
        method="auto",
        stride=1,
        smoothing_window=1,
        min_confidence=0.0,
        overlay=True,
        heatmap_strategies=["basic"],
    )
    assert artifacts.csv_path.exists()
    assert "basic" in artifacts.heatmaps
    assert artifacts.heatmaps["basic"].exists()
    assert len(artifacts.points) > 0
    if artifacts.overlay_video:
        assert Path(artifacts.overlay_video).exists()


def _make_dummy_video(path: Path) -> None:
    assert cv2 is not None and np is not None
    width, height = 320, 180
    fps = 15.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(60):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cx = int(40 + i * 3)
        cy = int(60 + 20 * np.sin(i / 10))
        cv2.circle(frame, (cx, cy), 6, (200, 180, 20), -1, lineType=cv2.LINE_AA)
        writer.write(frame)
    writer.release()
