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


def _make_stub(options: dict[str, Any]) -> AnalysisEngine:
    return StubEngine()


def _make_vlm(options: dict[str, Any]) -> AnalysisEngine:
    from .vlm import VlmClient, VlmEngine

    base_url = options.get("vlm_base_url")
    model = options.get("vlm_model")
    if not base_url or not model:
        raise ValueError(
            "the vlm engine needs an endpoint: pass --vlm-url and --vlm-model, "
            "or --vlm-config with --vlm-tier"
        )
    client = VlmClient(
        base_url=str(base_url),
        model=str(model),
        api_key=options.get("vlm_api_key"),
    )
    return VlmEngine(client, frame_count=int(options.get("vlm_frames", 6)))


def _make_detect(options: dict[str, Any]) -> AnalysisEngine:
    from .detect import DetectEngine, FrameDetector, UltralyticsDetector

    detector: FrameDetector
    injected = options.get("detector")
    if injected is not None:  # embedding/testing hook
        detector = injected
    else:
        detector = UltralyticsDetector(
            model_name=str(options.get("detect_model", "yolo11n.pt")),
            confidence=float(options.get("detect_conf", 0.35)),
        )
    return DetectEngine(
        detector,
        stride=int(options.get("detect_stride", 3)),
    )


ENGINES = {"stub": _make_stub, "vlm": _make_vlm, "detect": _make_detect}


def make_engine(name: str, options: dict[str, Any] | None = None) -> AnalysisEngine:
    try:
        factory = ENGINES[name]
    except KeyError as err:
        raise ValueError(
            f"unknown engine {name!r} — available: {', '.join(sorted(ENGINES))}"
        ) from err
    return factory(options or {})
