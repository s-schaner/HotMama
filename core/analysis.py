from __future__ import annotations

import base64
import math
import os
import tempfile
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt


__all__ = [
    "b64_image_data_uri",
    "extract_frames",
    "call_endpoint_openai_chat",
    "draw_court",
    "render_heatmap",
]


def b64_image_data_uri(frame, q=85) -> str:
    """Convert an OpenCV frame to a base64-encoded JPEG data URI."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, enc = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


def extract_frames(
    path: str, fps: float, max_frames: int
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dur = total / src_fps if src_fps > 0 else 0
    wanted = min(max_frames, int(math.ceil(dur * fps)) if dur > 0 else max_frames)
    wanted = max(1, wanted)

    if total > 0:
        idxs = np.linspace(0, max(total - 1, 0), num=wanted, dtype=int)
    else:
        idxs = np.arange(0, wanted, dtype=int)

    frames: List[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    return frames, {
        "src_fps": float(src_fps),
        "frames": len(frames),
        "duration": float(dur),
    }


def call_endpoint_openai_chat(
    endpoint: str,
    token: str | None,
    model: str,
    messages: List[Dict[str, Any]],
    fps: int,
    max_px: int,
    max_tokens: int,
    temperature: float,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible chat completion endpoint."""
    base = endpoint.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "mm_processor_kwargs": {"fps": int(fps), "max_pixels": int(max_px)},
    }

    response = requests.post(
        f"{base}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def draw_court(ax: plt.Axes) -> None:
    """Draw a volleyball court on the supplied axes."""
    ax.set_aspect("equal")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 18)
    ax.add_patch(plt.Rectangle((0, 0), 9, 18, fill=False, linewidth=2))
    ax.plot([0, 9], [9, 9], "k-", linewidth=1)
    ax.plot([0, 9], [6, 6], "k--", linewidth=1)
    ax.plot([0, 9], [12, 12], "k--", linewidth=1)
    ax.set_xlabel("Court X (m)")
    ax.set_ylabel("Court Y (m)")
    ax.set_title("Court (Top-Down)")
    ax.grid(False)


def render_heatmap(
    points3d: List[Tuple[float, float, float]], out_path: str | None
) -> str:
    """Render a heatmap for 3D points and return the saved path."""
    if not points3d:
        raise ValueError("No points")

    xs = np.array([p[0] for p in points3d], float)
    ys = np.array([p[1] for p in points3d], float)
    hs = np.array([p[2] for p in points3d], float)

    hmin, hmax = float(np.min(hs)), float(np.max(hs))
    if hmax - hmin < 1e-6:
        norm_h = np.zeros_like(hs)
    else:
        norm_h = (hs - hmin) / (hmax - hmin)

    fig, ax = plt.subplots(figsize=(6, 10))
    draw_court(ax)
    scatter = ax.scatter(
        xs,
        ys,
        c=norm_h,
        cmap=plt.cm.get_cmap("RdYlGn_r"),
        s=50,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.033, pad=0.02)
    colorbar.set_label("Height (normalized, green low → red high)")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tmp.name
