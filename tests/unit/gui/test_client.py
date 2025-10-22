from datetime import datetime
from uuid import uuid4

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from deploy.gui.app.client import ApiClient


def test_api_client_submit_status_and_download(tmp_path) -> None:
    job_id = uuid4()
    submitted_at = datetime.utcnow().isoformat()
    updated_at = datetime.utcnow().isoformat()
    artifact_bytes = b"video"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/jobs"):
            payload = request.json()
            assert payload["payload"]["task"] == "analyze_video"
            assert payload["payload"]["options"].get("overlays") == [
                "heatmap"
            ]
            return httpx.Response(
                202,
                json={
                    "job_id": str(job_id),
                    "status": "queued",
                    "submitted_at": submitted_at,
                    "profile": "cpu",
                    "priority": "normal",
                    "idempotency_key": "demo",
                },
            )
        if request.method == "GET" and request.url.path.endswith(f"/jobs/{job_id}"):
            return httpx.Response(
                200,
                json={
                    "job_id": str(job_id),
                    "status": "completed",
                    "submitted_at": submitted_at,
                    "updated_at": updated_at,
                    "profile": "cpu",
                    "message": None,
                    "artifact_path": "/app/sessions/example/result.mp4",
                    "priority": "high",
                },
            )
        if request.method == "GET" and request.url.path.endswith(
            f"/jobs/{job_id}/artifact"
        ):
            return httpx.Response(
                200,
                content=artifact_bytes,
                headers={"content-disposition": "attachment; filename=result.mp4"},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://testserver/v1") as http_client:
        client = ApiClient("http://testserver/v1", timeout=1.0, client=http_client)
        handle = client.submit_job(
            "/tmp/video.mp4",
            job_type="pipeline.analysis",
            overlays=["overlay.activity_heatmap"],
        )
        assert handle.job_id == job_id
        assert handle.status == "queued"
        assert handle.priority == "normal"
        assert handle.task == "analyze_video"
        assert handle.idempotency_key == "demo"

        state = client.get_status(job_id)
        assert state.status == "completed"
        assert state.as_dict()["artifact_path"] == "/app/sessions/example/result.mp4"
        assert state.priority == "high"

        artifact_path = client.download_artifact(job_id, tmp_path)
        assert artifact_path.name == "result.mp4"
        assert artifact_path.read_bytes() == artifact_bytes

        client.close()
