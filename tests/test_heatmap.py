from __future__ import annotations

from pathlib import Path

from viz.heatmap import render_heatmap


def test_render_heatmap(tmp_path: Path) -> None:
    points = [(1.0, 1.0, 0.1), (2.0, 5.0, 0.9)]
    out_path = tmp_path / "heatmap.png"
    result = render_heatmap(points, out_path)
    assert result.exists()
    assert result.stat().st_size > 0
