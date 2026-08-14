"""VLM engine: sampling, wire shape, parsing, tiers, probe — all offline."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("cv2")

from hotmama.worker.engine import make_engine  # noqa: E402
from hotmama.worker.vlm import (  # noqa: E402
    VlmClient,
    VlmEngine,
    VlmError,
    extract_json,
    load_tier_config,
    probe,
    sample_frames,
)

from ..capture.util import make_dummy_video  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID_SUMMARY = {
    "rally_visible": True,
    "description": "Serve from the far side, rally ends near the net.",
    "ball_landed": "near",
    "serve_visible": True,
    "jersey_numbers": [7, 12],
    "confidence": 0.72,
}


def _mock_client(handler: Any) -> VlmClient:
    return VlmClient(
        base_url="http://corona:8005",
        model="qwen3-vl-8b",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _completion(payload: Any) -> httpx.Response:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


class TestFrameSampling:
    def test_samples_include_first_and_last(self, tmp_path: Path) -> None:
        video = make_dummy_video(tmp_path / "clip.mp4", seconds=1.0, fps=20)  # 20 frames
        frames = sample_frames(video, count=6)
        assert len(frames) == 6
        raw = base64.b64decode(frames[0])
        assert raw[:2] == b"\xff\xd8"  # JPEG magic

    def test_short_clip_returns_all_frames(self, tmp_path: Path) -> None:
        video = make_dummy_video(tmp_path / "clip.mp4", seconds=0.2, fps=10)  # 2 frames
        assert len(sample_frames(video, count=6)) == 2

    def test_downscale_preserves_aspect(self, tmp_path: Path) -> None:
        import cv2
        import numpy as np

        video = make_dummy_video(tmp_path / "clip.mp4", seconds=0.5, fps=10, size=(160, 120))
        frames = sample_frames(video, count=2, max_edge=100)
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(frames[0]), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        height, width = image.shape[:2]
        assert (width, height) == (100, 75)  # 4:3 preserved

    def test_unreadable_clip_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.mp4"
        missing.write_bytes(b"not a video")
        with pytest.raises(VlmError):
            sample_frames(missing)


class TestExtractJson:
    def test_plain_and_fenced(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
        assert extract_json('Sure! Here you go: {"a": {"b": 2}} hope that helps') == {
            "a": {"b": 2}
        }

    def test_garbage_raises(self) -> None:
        with pytest.raises(VlmError):
            extract_json("no json here")
        with pytest.raises(VlmError):
            extract_json("[1, 2, 3]")


class TestVlmClient:
    def test_request_shape(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return _completion(VALID_SUMMARY)

        client = _mock_client(handler)
        text = client.describe_frames(["QUJD", "REVG"], "analyze this")
        assert json.loads(text) == VALID_SUMMARY
        assert seen["url"] == "http://corona:8005/v1/chat/completions"
        body = seen["body"]
        assert body["model"] == "qwen3-vl-8b"
        assert body["response_format"] == {"type": "json_object"}
        content = body["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "analyze this"}
        assert [part["type"] for part in content[1:]] == ["image_url", "image_url"]
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")

    def test_retries_without_response_format_on_400(self) -> None:
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "response_format" in body:
                return httpx.Response(400, text="response_format unsupported")
            return _completion(VALID_SUMMARY)

        client = _mock_client(handler)
        client.describe_frames(["QUJD"], "p")
        assert len(bodies) == 2
        assert "response_format" not in bodies[1]

    def test_server_error_raises(self) -> None:
        client = _mock_client(lambda request: httpx.Response(500, text="boom"))
        with pytest.raises(VlmError, match="500"):
            client.describe_frames(["QUJD"], "p")


class TestVlmEngine:
    def test_analyze_produces_observation(self, tmp_path: Path) -> None:
        video = make_dummy_video(tmp_path / "rally.mp4", seconds=1.0, fps=20)
        client = _mock_client(lambda request: _completion(VALID_SUMMARY))
        engine = VlmEngine(client, frame_count=4)

        observations = engine.analyze(video, {"clip_id": "c_1"})
        assert len(observations) == 1
        observation = observations[0]
        assert observation["kind"] == "vlm_rally_summary"
        assert observation["confidence"] == 0.72
        assert observation["data"]["jersey_numbers"] == [7, 12]
        assert observation["data"]["ball_landed"] == "near"
        assert observation["data"]["frames_analyzed"] == 4
        assert "confidence" not in observation["data"]

    def test_invalid_model_json_raises(self, tmp_path: Path) -> None:
        video = make_dummy_video(tmp_path / "rally.mp4", seconds=0.5, fps=10)
        client = _mock_client(
            lambda request: _completion({"rally_visible": "definitely"})
        )
        engine = VlmEngine(client)
        with pytest.raises(VlmError, match="validation"):
            engine.analyze(video, {})


class TestTiersAndFactory:
    def test_example_config_loads(self) -> None:
        tiers = load_tier_config(REPO_ROOT / "examples" / "vlm-tiers.example.json")
        assert set(tiers) == {"deep", "standard", "fast"}
        assert tiers["deep"].base_url == "http://corona:8000"
        assert tiers["deep"].model == "qwen3-vl-30b"
        assert tiers["fast"].max_frames == 4

    def test_bad_config_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"nope": true}')
        with pytest.raises(VlmError, match="tiers"):
            load_tier_config(bad)

    def test_factory_requires_endpoint(self) -> None:
        with pytest.raises(ValueError, match="endpoint"):
            make_engine("vlm", {})
        engine = make_engine(
            "vlm", {"vlm_base_url": "http://corona:8004", "vlm_model": "qwen3-vl-4b"}
        )
        assert engine.name == "vlm:qwen3-vl-4b"


class TestProbe:
    def test_probe_reports_vision_ok(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["messages"][0]["content"][1]["type"] == "image_url"
            return _completion({"shape": "circle", "color": "red"})

        result = probe(_mock_client(handler))
        assert result["vision_ok"] is True
        assert result["model"] == "qwen3-vl-8b"
        assert "latency_s" in result

    def test_probe_flags_wrong_answer(self) -> None:
        result = probe(
            _mock_client(lambda request: _completion({"shape": "square", "color": "blue"}))
        )
        assert result["vision_ok"] is False
