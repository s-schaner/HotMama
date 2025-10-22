from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from deploy.api.app.dependencies import get_job_parser, get_redis
from deploy.api.app.main import create_app
from deploy.api.app.schemas import JobPayload, JobSpec


class FakeRedis:
    def __init__(self) -> None:
        self.queue: list[bytes] = []
        self.hashes: dict[str, dict[str, str]] = {}

    def rpush(self, key: str, value: str) -> None:
        self.queue.append(value.encode())

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        store = self.hashes.setdefault(key, {})
        store.update(mapping)

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        data = self.hashes.get(key, {})
        return {k.encode(): v.encode() for k, v in data.items()}

    def expire(self, key: str, ttl: int) -> None:  # pragma: no cover - noop
        self.hashes.setdefault(key, {})["_ttl"] = str(ttl)

    def close(self) -> None:  # pragma: no cover - noop
        pass


class StubParser:
    def __init__(self, spec: JobSpec) -> None:
        self.spec = spec
        self.calls: list[str] = []

    def parse(self, text: str) -> JobSpec:
        self.calls.append(text)
        return self.spec


@contextmanager
def patched_app(fake: FakeRedis, parser: StubParser | None = None) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_redis] = lambda: fake
    if parser is not None:
        app.dependency_overrides[get_job_parser] = lambda: parser

    from deploy.api.app import main as main_module

    original_from_url = main_module.redis.from_url
    main_module.redis.from_url = lambda *_, **__: fake  # type: ignore
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_job_parser, None)
        app.dependency_overrides.pop(get_redis, None)
        main_module.redis.from_url = original_from_url


def test_submit_and_fetch_job() -> None:
    fake = FakeRedis()
    with patched_app(fake) as client:
        response = client.post(
            "/v1/jobs",
            json={
                "payload": {
                    "task": "analyze_video",
                    "source_uri": "/tmp/example.mp4",
                    "clips": [
                        {"start": "00:00:01.0", "end": "00:00:03.250"}
                    ],
                    "options": {"mode": "fast"},
                }
            },
        )
        assert response.status_code == 202
        body = response.json()
        job_id = UUID(body["job_id"])
        assert body["priority"] == "normal"
        assert body["idempotency_key"] is not None
        assert fake.queue, "job should be enqueued"

        queued = json.loads(fake.queue[0].decode())
        assert queued["priority"] == "normal"
        assert queued["payload"]["options"] == {"mode": "fast"}
        assert queued["payload"]["clips"] == [
            {"start": "00:00:01", "end": "00:00:03.25"}
        ]

        status = client.get(f"/v1/jobs/{job_id}")
        assert status.status_code == 200
        status_body = status.json()
        assert status_body["status"] == "queued"
        assert UUID(status_body["job_id"]) == job_id
        assert status_body["priority"] == "normal"

        artifact = client.get(f"/v1/jobs/{job_id}/artifact")
        assert artifact.status_code == 404


def test_submit_job_via_natural_language() -> None:
    fake = FakeRedis()
    spec = JobSpec(
        payload=JobPayload(
            task="detect_events",
            source_uri="example.mov",
            options={"sensitivity": "high"},
        ),
        priority="high",
        idempotency_key="abc123",
    )
    parser = StubParser(spec)

    with patched_app(fake, parser=parser) as client:
        response = client.post(
            "/v1/jobs",
            json={"nl": "Detect events in example.mov with high sensitivity"},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["priority"] == "high"
        assert body["idempotency_key"] == "abc123"
        assert parser.calls == [
            "Detect events in example.mov with high sensitivity"
        ]

        queued = json.loads(fake.queue[0].decode())
        assert queued["priority"] == "high"
        assert queued["payload"]["task"] == "detect_events"
        assert queued["payload"]["options"] == {"sensitivity": "high"}
