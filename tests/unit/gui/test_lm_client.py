import json

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from deploy.gui.app.client import LMStudioClient


def test_lmstudio_client_generate_manifest_with_enrichment() -> None:
    manifest = {
        "payload": {
            "task": "analyze_video",
            "source_uri": "/tmp/video.mp4",
            "options": {"threshold": 0.4},
        },
        "priority": "normal",
    }
    enriched = {
        "payload": {
            "task": "analyze_video",
            "source_uri": "/tmp/video.mp4",
            "options": {"threshold": 0.4, "overlays": ["pose"]},
        },
        "priority": "normal",
    }

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = request.json()
        calls.append(data["model"])
        if data["model"] == "qwen2.5-3b-instruct":
            content = json.dumps(manifest)
        else:
            content = json.dumps(enriched)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://lmstudio") as http_client:
        client = LMStudioClient(
            "http://lmstudio",
            api_key="demo",
            instruct_model="qwen2.5-3b-instruct",
            enrichment_model="qwen2.5-vl-7b",
            client=http_client,
        )
        spec = client.generate_manifest(
            "Analyse the clip",
            source_uri="/tmp/video.mp4",
            task_hint="analyze_video",
            enrich=True,
        )

    assert calls == ["qwen2.5-3b-instruct", "qwen2.5-vl-7b"]
    assert spec.model_dump(mode="json") == enriched


def test_lmstudio_client_raises_on_invalid_payload() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://lmstudio") as http_client:
        client = LMStudioClient(
            "http://lmstudio",
            api_key="demo",
            instruct_model="qwen2.5-3b-instruct",
            client=http_client,
        )
        with pytest.raises(RuntimeError):
            client.generate_manifest(
                "Describe the task",
                source_uri="/tmp/video.mp4",
                task_hint="analyze_video",
            )
