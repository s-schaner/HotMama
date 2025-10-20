from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterable, Literal, Tuple

try:  # pragma: no cover - optional heavy dependencies
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy dependencies
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ModuleNotFoundError:  # pragma: no cover
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    LinearSegmentedColormap = None  # type: ignore[assignment]

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    _HAS_CV2 = False

COURT_WIDTH = 9
COURT_LENGTH = 18
CANVAS_SCALE = 40

COURT_W = float(COURT_WIDTH)
COURT_L = float(COURT_LENGTH)

_HAS_SEXY_DEPS = all(obj is not None for obj in (np, matplotlib, plt, LinearSegmentedColormap))


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


def _ensure_sexy_dependencies() -> None:
    if not _HAS_SEXY_DEPS:
        raise RuntimeError("render_heatmap_sexy requires NumPy and Matplotlib")


def _draw_court(ax, theme: Literal["dark", "light"] = "dark") -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0, COURT_W)
    ax.set_ylim(0, COURT_L)
    if theme == "dark":
        fg = "#e5e7eb"
        grid = "#9ca3af"
        bg = "#111827"
        ax.set_facecolor(bg)
    else:
        fg = "#111827"
        grid = "#6b7280"
        bg = "#ffffff"
        ax.set_facecolor(bg)

    ax.add_patch(
        plt.Rectangle((0, 0), COURT_W, COURT_L, fill=False, linewidth=2, edgecolor=fg)
    )
    ax.plot([0, COURT_W], [COURT_L / 2, COURT_L / 2], color=fg, linewidth=1.2)
    ax.plot([0, COURT_W], [6, 6], color=grid, linestyle="--", linewidth=1)
    ax.plot([0, COURT_W], [12, 12], color=grid, linestyle="--", linewidth=1)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _make_colormap(name: str) -> matplotlib.colors.Colormap:
    presets = {
        "plasma": plt.cm.plasma,
        "magma": plt.cm.magma,
        "inferno": plt.cm.inferno,
        "turbo": plt.cm.turbo,
        "mintred": LinearSegmentedColormap.from_list(
            "mintred", ["#00f5d4", "#00bbf9", "#3a86ff", "#ff006e", "#ffd166"]
        ),
    }
    return presets.get(name, plt.cm.magma)


def _rasterize_points(points: np.ndarray, res: int = 140) -> Tuple[np.ndarray, float, float]:
    px_w = int(COURT_W * res)
    px_l = int(COURT_L * res)
    grid = np.zeros((px_l, px_w), dtype=np.float32)

    xs = np.clip((points[:, 0] * res).round().astype(int), 0, px_w - 1)
    ys = np.clip((points[:, 1] * res).round().astype(int), 0, px_l - 1)

    grid[ys, xs] += 1.0
    sigma_base = max(2.0, res * 0.06)
    return grid, float(res), float(sigma_base)


def _blur(grid: np.ndarray, sigma: float) -> np.ndarray:
    if _HAS_CV2:
        k = int(max(3, round(sigma * 4)) // 2 * 2 + 1)
        return cv2.GaussianBlur(
            grid, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
        )

    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1)
    g = np.exp(-(x ** 2) / (2 * sigma * sigma))
    g /= g.sum()
    tmp = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, grid)
    out = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, tmp)
    return out


def _normalize(array: np.ndarray) -> np.ndarray:
    maximum = array.max()
    if maximum <= 1e-8:
        return array
    return array / maximum


def _draw_trail(
    ax, points3d: np.ndarray, cmap: matplotlib.colors.Colormap, lw_px: float, glow: bool = True
) -> None:
    hs = points3d[:, 2]
    if np.nanmax(hs) - np.nanmin(hs) < 1e-9:
        hn = np.zeros_like(hs)
    else:
        hn = (hs - np.nanmin(hs)) / (np.nanmax(hs) - np.nanmin(hs))

    if glow:
        for g_mul, alpha in [(3.0, 0.12), (2.0, 0.18)]:
            ax.plot(
                points3d[:, 0],
                points3d[:, 1],
                linewidth=lw_px * g_mul / 72,
                color=(1, 1, 1, alpha),
                solid_capstyle="round",
                zorder=8,
            )

    for i in range(len(points3d) - 1):
        color = cmap(hn[i])
        ax.plot(
            points3d[i : i + 2, 0],
            points3d[i : i + 2, 1],
            linewidth=lw_px / 72,
            color=color,
            solid_capstyle="round",
            zorder=9,
        )


def render_heatmap_sexy(
    points: Iterable[Tuple[float, float, float]],
    output_path: Path | str,
    theme: Literal["dark", "light"] = "dark",
    colormap: str = "magma",
    dpi: int = 220,
    res: int = 140,
    density_scale: float = 1.0,
    density_sigma_mul: float = 1.0,
    show_contours: bool = True,
    draw_trail: bool = True,
    trail_width_px: float = 10.0,
) -> Path:
    _ensure_sexy_dependencies()
    pts = np.array(list(points), dtype=float)
    if len(pts) == 0:
        raise ValueError("No points for sexy heatmap")

    fig_w, fig_h = 6, 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    _draw_court(ax, theme=theme)
    cmap = _make_colormap(colormap)

    dens_grid, _, sigma_base = _rasterize_points(pts[:, :2], res=res)
    dens_blur = _blur(dens_grid, sigma=sigma_base * float(density_sigma_mul))
    dens = _normalize(dens_blur) ** 0.9
    dens *= float(density_scale)

    extent = [0, COURT_W, 0, COURT_L]
    dens_rgba = cmap(dens)
    dens_rgba[..., 3] = np.clip(dens, 0.0, 0.92)
    ax.imshow(
        dens_rgba,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        zorder=6,
    )

    if show_contours:
        levels = np.linspace(0.2, 0.9, 4)
        ax.contour(
            dens,
            levels=levels,
            colors="#ffffff",
            alpha=0.15,
            linewidths=0.8,
            origin="lower",
            extent=extent,
            zorder=7,
        )

    if draw_trail and len(pts) >= 2:
        _draw_trail(ax, pts, cmap, lw_px=trail_width_px, glow=True)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(ax, width="35%", height="3%", loc="upper right", borderpad=1.0)
    grad = np.linspace(0, 1, 256).reshape(1, 256)
    cax.imshow(grad, aspect="auto", cmap=cmap, origin="lower")
    cax.set_xticks([0, 128, 255])
    cax.set_xticklabels(["low h", "mid", "high h"])
    cax.set_yticks([])
    cax.set_frame_on(False)

    ax.set_title(
        "Ball Trajectory Density + Height Trail",
        fontsize=10,
        color="#e5e7eb" if theme == "dark" else "#111827",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
