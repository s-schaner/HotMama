"""Worker runtime loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import redis

from .config import Settings, get_settings
from .processor import VisionProcessor
from .queue import WorkerQueue
from .schemas import ProcessResult, QueuedJob

LOGGER = logging.getLogger("hotmama.worker")


class Worker:
    """Simple Redis-backed worker."""

    def __init__(
        self,
        redis_client_factory: Callable[[], redis.Redis] | None = None,
        settings: Settings | None = None,
        processor: VisionProcessor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._redis_factory = redis_client_factory or (
            lambda: redis.from_url(self._settings.redis_url, decode_responses=False)
        )
        self._processor = processor or VisionProcessor(self._settings)
        self._queue: WorkerQueue | None = None

    def _ensure_queue(self) -> WorkerQueue:
        if self._queue is None:
            client = self._redis_factory()
            self._queue = WorkerQueue(client, self._settings)
        return self._queue

    def run_forever(self) -> None:
        log_format = (
            "%(message)s"
            if self._settings.log_json
            else "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        logging.basicConfig(level=logging.INFO, format=log_format)
        queue = self._ensure_queue()
        LOGGER.info(
            "worker started",
            extra={
                "service": self._settings.service_name,
                "profile": self._settings.deployment_profile,
            },
        )
        backoff = 1
        while True:
            try:
                for job in queue.listen():
                    self._handle_job(queue, job)
                    backoff = 1
            except redis.RedisError as exc:  # pragma: no cover - network failure path
                LOGGER.exception("redis error", extra={"error": str(exc)})
                time.sleep(backoff)
                backoff = min(backoff * 2, self._settings.max_backoff_seconds)

    def _handle_job(self, queue: WorkerQueue, job: QueuedJob) -> None:
        LOGGER.info("processing job", extra={"job_id": str(job.job_id)})
        queue.mark_status(job, "processing")
        try:
            result = self._processor.process(job)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception(
                "job failed",
                extra={"job_id": str(job.job_id), "error": str(exc)},
            )
            result = ProcessResult(status="failed", message=str(exc))
        queue.finish(job, result)
        LOGGER.info(
            "job completed",
            extra={
                "job_id": str(job.job_id),
                "status": result.status,
                "artifact": result.artifact_path,
            },
        )
