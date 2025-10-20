from __future__ import annotations

from pathlib import Path
from typing import Iterable

from viz.heatmap import _write_png


def draw_tracks(image_path: Path, tracks: Iterable[dict], output_path: Path) -> Path:
    width, height = 640, 360
    background = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    for idx, track in enumerate(tracks):
        x = int(track.get("x", (idx + 1) * 10) % width)
        y = int(track.get("y", (idx + 1) * 15) % height)
        if 0 <= x < width and 0 <= y < height:
            background[y][x] = (255, 0, 0)
    _write_png(background, output_path)
    return output_path
