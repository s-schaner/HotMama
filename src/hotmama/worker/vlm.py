"""VLM analysis engine: rally chunk → sampled frames → vision model → observations.

Speaks the OpenAI-compatible vision chat shape vLLM serves (data-URL image
parts), so any tier on the inference network works: point it at a 4B for
fast triage or the 30B for depth — the endpoint is operator configuration,
never code (D15/D16).

Observations are deliberately attribution-free ("ball landed near/far",
"jerseys 7 and 12 visible") — mapping court sides to teams requires the
camera calibration phase, and the engine must not guess what it cannot know.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

LOGGER = logging.getLogger("hotmama.worker.vlm")


class VlmError(RuntimeError):
    pass


# -- frame sampling ------------------------------------------------------------


def sample_frames(
    clip_path: Path, *, count: int = 6, max_edge: int = 768, jpeg_quality: int = 85
) -> list[str]:
    """Uniformly sample base64 JPEG frames, always including first and last.

    (The v1 codebase truncated to 3 frames and never sampled the final frame —
    the moment volleyball rallies are decided. Both fixed here.)
    """
    try:
        import cv2
    except ImportError as err:
        raise VlmError("the vlm engine needs OpenCV — install hotmama[capture]") from err

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        capture.release()
        raise VlmError(f"could not open clip {clip_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            frames = _read_all(capture)
            total = len(frames)
            if total == 0:
                raise VlmError("clip contains no decodable frames")
            picks = _indices(total, count)
            selected = [frames[i] for i in picks]
        else:
            picks = _indices(total, count)
            selected = []
            for index in picks:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = capture.read()
                if success:
                    selected.append(frame)
            if not selected:
                raise VlmError("clip contains no decodable frames")

        encoded: list[str] = []
        for frame in selected:
            height, width = frame.shape[:2]
            scale = max_edge / max(height, width)
            if scale < 1.0:
                frame = cv2.resize(
                    frame, (max(1, round(width * scale)), max(1, round(height * scale)))
                )
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if not ok:
                raise VlmError("jpeg encoding failed")
            encoded.append(base64.b64encode(buffer.tobytes()).decode("ascii"))
        return encoded
    finally:
        capture.release()


def _read_all(capture: Any) -> list[Any]:
    frames: list[Any] = []
    while True:
        success, frame = capture.read()
        if not success:
            return frames
        frames.append(frame)


def _indices(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    step = (total - 1) / (count - 1)
    return sorted({round(i * step) for i in range(count)})


# -- structured output ---------------------------------------------------------


class VlmRallySummary(BaseModel):
    """What the model must return. ``extra=forbid`` keeps hallucinated keys out."""

    model_config = ConfigDict(extra="forbid")

    rally_visible: bool
    description: str = Field(max_length=600)
    ball_landed: Literal["near", "far", "out_of_view", "unknown"] = "unknown"
    serve_visible: bool = False
    jersey_numbers: list[int] = Field(default_factory=list, max_length=24)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_PROMPT = """You are analyzing {count} frames sampled uniformly (in order) from one
volleyball rally clip filmed from a fixed end-of-court camera. "near" means the court
half closest to the camera, "far" the opposite half. Reply with ONLY a JSON object:
{{"rally_visible": bool, "description": str (<=2 sentences),
"ball_landed": "near"|"far"|"out_of_view"|"unknown", "serve_visible": bool,
"jersey_numbers": [visible jersey numbers as integers],
"confidence": 0.0-1.0 (your overall confidence)}}.
Never attribute points to teams; only report what is visible.""".strip()


def extract_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction: strip fences, find the outermost object."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise VlmError(f"no JSON object in model output: {text[:120]!r}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as err:
        raise VlmError(f"invalid JSON from model: {err}") from err
    if not isinstance(parsed, dict):
        raise VlmError("model output JSON is not an object")
    return parsed


# -- the client and engine -----------------------------------------------------


@dataclass(frozen=True)
class VlmTier:
    name: str
    base_url: str
    model: str
    max_frames: int = 6

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> VlmTier:
        return cls(
            name=name,
            base_url=str(data["base_url"]).rstrip("/"),
            model=str(data["model"]),
            max_frames=int(data.get("max_frames", 6)),
        )


def load_tier_config(path: Path) -> dict[str, VlmTier]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise VlmError(f"could not read tier config {path}: {err}") from err
    tiers = raw.get("tiers") if isinstance(raw, dict) else None
    if not isinstance(tiers, dict) or not tiers:
        raise VlmError(f'tier config {path} must contain a non-empty "tiers" object')
    return {name: VlmTier.from_dict(name, data) for name, data in tiers.items()}


class VlmClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 700,
        temperature: float = 0.1,
        http: httpx.Client | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self._url = (
            f"{base}/chat/completions"
            if base.endswith("/v1")
            else f"{base}/v1/chat/completions"
        )
        self.model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._http = http or httpx.Client(timeout=180.0)

    def describe_frames(self, frames_b64: list[str], prompt: str) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
            for frame in frames_b64
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
        }
        response = self._post(body)
        if response.status_code == 400:
            # Older vLLM builds reject response_format for VLMs; the prompt
            # already demands JSON, so retry bare.
            body.pop("response_format")
            response = self._post(body)
        if response.status_code != 200:
            raise VlmError(
                f"vision endpoint returned {response.status_code}: {response.text[:200]}"
            )
        try:
            content_text = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as err:
            raise VlmError("malformed completion from vision endpoint") from err
        if not isinstance(content_text, str) or not content_text.strip():
            raise VlmError("empty completion from vision endpoint")
        return content_text

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            return self._http.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as err:
            raise VlmError(f"vision endpoint unreachable: {err}") from err


class VlmEngine:
    """AnalysisEngine backed by a vision-language tier."""

    def __init__(self, client: VlmClient, *, frame_count: int = 6) -> None:
        self.name = f"vlm:{client.model}"
        self._client = client
        self._frame_count = max(2, frame_count)

    def analyze(self, clip_path: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
        frames = sample_frames(clip_path, count=self._frame_count)
        prompt = _PROMPT.format(count=len(frames))
        raw_text = self._client.describe_frames(frames, prompt)
        try:
            summary = VlmRallySummary.model_validate(extract_json(raw_text))
        except ValidationError as err:
            raise VlmError(f"model JSON failed validation: {err.errors()[:2]}") from err
        return [
            {
                "kind": "vlm_rally_summary",
                "data": {
                    "engine": self.name,
                    "frames_analyzed": len(frames),
                    **summary.model_dump(exclude={"confidence"}),
                },
                "confidence": summary.confidence,
            }
        ]


def probe(client: VlmClient) -> dict[str, Any]:
    """Connectivity + vision sanity check: one synthetic image, one question."""
    try:
        import cv2
        import numpy as np
    except ImportError as err:
        raise VlmError("probe needs OpenCV — install hotmama[capture]") from err

    canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.circle(canvas, (128, 128), 80, (0, 0, 255), -1)  # BGR: a red circle
    ok, buffer = cv2.imencode(".jpg", canvas)
    if not ok:
        raise VlmError("could not encode probe image")
    frame = base64.b64encode(buffer.tobytes()).decode("ascii")

    started = time.monotonic()
    text = client.describe_frames(
        [frame],
        'Reply with ONLY JSON: {"shape": str, "color": str} describing the image.',
    )
    elapsed = time.monotonic() - started
    answer = extract_json(text)
    looks_right = "circle" in str(answer.get("shape", "")).lower() and "red" in str(
        answer.get("color", "")
    ).lower()
    return {
        "model": client.model,
        "latency_s": round(elapsed, 2),
        "answer": answer,
        "vision_ok": looks_right,
    }
