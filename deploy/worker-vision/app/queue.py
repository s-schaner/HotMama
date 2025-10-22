"""Redis helpers for the worker."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from uuid import UUID

import redis

from .config import Settings, get_settings
from .schemas import ProcessResult, QueuedJob


class WorkerQueue:
    def __init__(self, client: redis.Redis, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()

    @property
    def queue_name(self) -> str:
        return self._settings.redis_queue_name

    def _status_key(self, job_id: UUID) -> str:
        return f"{self._settings.redis_status_prefix}:{job_id}"

    def listen(self) -> Iterator[QueuedJob]:
        while True:
            result = self._client.blpop(
                self.queue_name, timeout=self._settings.poll_timeout
            )
            if result is None:
                continue
            _, payload = result
            job = QueuedJob.model_validate_json(payload)
            yield job

    def mark_status(
        self, job: QueuedJob, status: str, message: str | None = None
    ) -> None:
        now = datetime.utcnow().isoformat()
        mapping: dict[str, str] = {"status": status, "updated_at": now}
        if message:
            mapping["message"] = message
        self._client.hset(self._status_key(job.job_id), mapping=mapping)

    def finish(self, job: QueuedJob, result: ProcessResult) -> None:
        mapping = {
            "status": result.status,
            "updated_at": datetime.utcnow().isoformat(),
            "message": result.message,
        }
        if result.artifact_path:
            mapping["artifact_path"] = result.artifact_path
        self._client.hset(self._status_key(job.job_id), mapping=mapping)
