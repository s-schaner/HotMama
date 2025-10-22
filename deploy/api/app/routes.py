"""API route definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from redis import Redis

from .config import Settings, get_settings
from .dependencies import get_job_parser, get_redis
from .queue import RedisQueue
from .parsing import JobParser, JobParserError
from .schemas import (
    HealthResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobStatus,
)

router = APIRouter()


class Dependencies:
    """Bundle runtime dependencies for easy injection."""

    def __init__(
        self, redis_client: Redis, settings: Settings, parser: JobParser
    ) -> None:
        self.redis_client = redis_client
        self.settings = settings
        self.queue = RedisQueue(redis_client, settings)
        self.parser = parser


async def get_dependencies(
    settings: Annotated[Settings, Depends(get_settings)],
    redis_client: Annotated[Redis, Depends(get_redis)],
    parser: Annotated[JobParser, Depends(get_job_parser)],
) -> Dependencies:
    return Dependencies(redis_client, settings, parser)


@router.get("/healthz", response_model=HealthResponse, tags=["system"])
async def healthz(
    settings: Annotated[Settings, Depends(get_settings)]
) -> HealthResponse:
    """Simple health-check endpoint."""

    return HealthResponse(service=settings.service_name)


@router.post("/jobs", response_model=JobCreateResponse, status_code=202, tags=["jobs"])
async def submit_job(
    request: JobCreateRequest,
    deps: Annotated[Dependencies, Depends(get_dependencies)],
) -> JobCreateResponse:
    """Submit a new job into the queue."""

    try:
        spec = request.to_spec(deps.parser)
    except JobParserError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if spec.idempotency_key is None:
        spec = spec.model_copy(update={"idempotency_key": str(uuid4())})

    job = JobCreateResponse(
        job_id=uuid4(),
        profile=deps.settings.deployment_profile,
        priority=spec.priority,
        idempotency_key=spec.idempotency_key,
    )
    deps.queue.enqueue(job, spec)
    return job


@router.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
async def job_status(
    job_id: UUID,
    deps: Annotated[Dependencies, Depends(get_dependencies)],
) -> JobStatus:
    status = deps.queue.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@router.get("/jobs/{job_id}/artifact", tags=["jobs"])
async def job_artifact(
    job_id: UUID,
    deps: Annotated[Dependencies, Depends(get_dependencies)],
) -> Response:
    status = deps.queue.get_status(job_id)
    if not status or not status.artifact_path:
        raise HTTPException(status_code=404, detail="artifact not ready")
    artifact_path = Path(status.artifact_path)
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="artifact missing")
    return FileResponse(
        artifact_path,
        filename=artifact_path.name,
        media_type="application/octet-stream",
        headers={
            "x-job-id": str(job_id),
            "x-finished-at": status.updated_at.isoformat(),
        },
    )
