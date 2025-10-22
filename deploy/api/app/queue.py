"""Redis queue helpers used by the API service."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import redis

from .config import Settings, get_settings
from .schemas import JobCreateResponse, JobStatus


class RedisQueue:
    """Thin wrapper that abstracts queue operations."""

    def __init__(self, client: redis.Redis, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()

    @property
    def queue_name(self) -> str:
        return self._settings.redis_queue_name

    def _status_key(self, job_id: UUID) -> str:
        return f"{self._settings.redis_status_prefix}:{job_id}"

    def enqueue(self, job: JobCreateResponse, payload: dict[str, Any]) -> None:
        """Store metadata and push the job onto the queue."""

        job_data = {
            "job_id": str(job.job_id),
            "submitted_at": job.submitted_at.isoformat(),
            "status": job.status,
            "profile": job.profile,
            "payload": payload,
        }
        self._client.rpush(self.queue_name, json.dumps(job_data))
        self._client.hset(
            self._status_key(job.job_id),
            mapping={
                "status": job.status,
                "submitted_at": job.submitted_at.isoformat(),
                "updated_at": job.submitted_at.isoformat(),
                "profile": job.profile,
            },
        )
        self._client.expire(
            self._status_key(job.job_id), self._settings.result_ttl_seconds
        )

    def get_status(self, job_id: UUID) -> JobStatus | None:
        """Fetch the current status if present."""

        data = self._client.hgetall(self._status_key(job_id))
        if not data:
            return None
        decoded = {key.decode(): value.decode() for key, value in data.items()}
        return JobStatus(
            job_id=job_id,
            status=decoded.get("status", "queued"),
            submitted_at=datetime.fromisoformat(decoded["submitted_at"]),
            updated_at=datetime.fromisoformat(decoded["updated_at"]),
            message=decoded.get("message"),
            artifact_path=decoded.get("artifact_path"),
            profile=decoded.get("profile", "cpu"),
        )

    def mark_complete(
        self,
        job_id: UUID,
        status: str,
        message: str | None = None,
        artifact_path: str | None = None,
    ) -> None:
        """Update status for job completion or failure."""

        now = datetime.utcnow().isoformat()
        mapping = {
            "status": status,
            "updated_at": now,
        }
        if message:
            mapping["message"] = message
        if artifact_path:
            mapping["artifact_path"] = artifact_path
        self._client.hset(self._status_key(job_id), mapping=mapping)
        self._client.expire(
            self._status_key(job_id), self._settings.result_ttl_seconds
        )
