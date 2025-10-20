from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterable, Tuple

COURT_WIDTH = 9
COURT_LENGTH = 18
CANVAS_SCALE = 40


def draw_court_background(width: int, height: int) -> list[list[Tuple[int, int, int]]]:
    background_color = (13, 59, 102)
    canvas = [[background_color for _ in range(width)] for _ in range(height)]
    mid_y = height // 2
    for x in range(width):
        canvas[mid_y][x] = (255, 255, 255)
    for y_meter in (6, 12):
        y = int(y_meter * CANVAS_SCALE)
        if 0 <= y < height:
            for x in range(width):
                canvas[y][x] = (200, 200, 200)
    return canvas


def _color_for_height(height: float, min_h: float, max_h: float) -> Tuple[int, int, int]:
    if max_h == min_h:
        ratio = 0.0
    else:
        ratio = (height - min_h) / (max_h - min_h)
    r = int(34 + ratio * (255 - 34))
    g = int(180 - ratio * 120)
    b = int(34 + (1 - ratio) * 34)
    return r, max(0, min(255, g)), b


def _draw_circle(canvas, cx: int, cy: int, radius: int, color: Tuple[int, int, int]) -> None:
    height = len(canvas)
    width = len(canvas[0])
    for y in range(cy - radius, cy + radius + 1):
        if y < 0 or y >= height:
            continue
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or x >= width:
                continue
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                canvas[y][x] = color


def _write_png(canvas: list[list[Tuple[int, int, int]]], path: Path) -> None:
    height = len(canvas)
    width = len(canvas[0])
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_rows = []
    for row in canvas:
        raw = bytearray()
        for r, g, b in row:
            raw.extend([r, g, b])
        raw_rows.append(b"\x00" + bytes(raw))
    idat = zlib.compress(b"".join(raw_rows), 9)
    png = signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(png)


def render_heatmap(points: Iterable[Tuple[float, float, float]], output_path: Path) -> Path:
    pts = list(points)
    if not pts:
        raise ValueError("No points provided for heatmap")
    width = COURT_WIDTH * CANVAS_SCALE
    height = COURT_LENGTH * CANVAS_SCALE
    canvas = draw_court_background(width, height)
    heights = [h for _, _, h in pts]
    min_h, max_h = min(heights), max(heights)
    for x, y, h in pts:
        cx = int(x * CANVAS_SCALE)
        cy = int(y * CANVAS_SCALE)
        color = _color_for_height(h, min_h, max_h)
        _draw_circle(canvas, cx, cy, radius=8, color=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(canvas, output_path)
    return output_path
