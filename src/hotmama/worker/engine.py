"""Analysis engines: a rally chunk in, observations out.

The engine is the swap point for real models. A box in the inference network
runs `hotmama-worker --engine <name>`; everything else (lease, download,
result posting, retries) is identical regardless of what analyzes the clip.

Observation shape (matches the worker feed's schema):
    {"kind": str, "data": {...}, "confidence": 0..1}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class AnalysisEngine(Protocol):
    name: str

    def analyze(self, clip_path: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
        """Return observations for one rally chunk. Blocking is fine."""
        ...


class StubEngine:
    """Proves the loop end to end: decodes the chunk, reports basic stats.

    Real CV engines (detection, tracking, OCR) replace this without touching
    the transport. Emits `clip_stats` when OpenCV can decode the clip, and a
    bare `clip_received` otherwise.
    """

    name = "stub"

    def analyze(self, clip_path: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
        size = clip_path.stat().st_size
        try:
            import cv2
        except ImportError:
            return [
                {
                    "kind": "clip_received",
                    "data": {"bytes": size, "engine": self.name},
                    "confidence": 1.0,
                }
            ]

        capture = cv2.VideoCapture(str(clip_path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = 0
            brightness_total = 0.0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frames += 1
                if frames % 10 == 1:  # sample, don't grind
                    brightness_total += float(frame.mean())
            samples = (frames + 9) // 10
            return [
                {
                    "kind": "clip_stats",
                    "data": {
                        "engine": self.name,
                        "bytes": size,
                        "frames": frames,
                        "fps": fps,
                        "duration_s": round(frames / fps, 2) if fps > 0 else None,
                        "brightness_mean": round(brightness_total / samples, 1)
                        if samples
                        else None,
                    },
                    "confidence": 1.0,
                }
            ]
        finally:
            capture.release()


ENGINES: dict[str, type] = {"stub": StubEngine}


def make_engine(name: str) -> AnalysisEngine:
    try:
        engine_cls = ENGINES[name]
    except KeyError as err:
        raise ValueError(
            f"unknown engine {name!r} — available: {', '.join(sorted(ENGINES))}"
        ) from err
    engine: AnalysisEngine = engine_cls()
    return engine
