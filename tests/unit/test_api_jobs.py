from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from fastapi.testclient import TestClient

from deploy.api.app.dependencies import get_redis
from deploy.api.app.main import create_app


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


@contextmanager
def patched_app(fake: FakeRedis) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_redis] = lambda: fake

    from deploy.api.app import main as main_module

    original_from_url = main_module.redis.from_url
    main_module.redis.from_url = lambda *_, **__: fake  # type: ignore
    try:
        with TestClient(app) as client:
            yield client
    finally:
        main_module.redis.from_url = original_from_url


def test_submit_and_fetch_job() -> None:
    fake = FakeRedis()
    with patched_app(fake) as client:
        response = client.post(
            "/v1/jobs",
            json={"payload": {"source_uri": "/tmp/example.mp4", "parameters": {}}},
        )
        assert response.status_code == 202
        body = response.json()
        job_id = UUID(body["job_id"])
        assert fake.queue, "job should be enqueued"

        status = client.get(f"/v1/jobs/{job_id}")
        assert status.status_code == 200
        status_body = status.json()
        assert status_body["status"] == "queued"
        assert UUID(status_body["job_id"]) == job_id

        artifact = client.get(f"/v1/jobs/{job_id}/artifact")
        assert artifact.status_code == 404
