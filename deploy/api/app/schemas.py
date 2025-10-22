"""Pydantic models shared by API components."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobPayload(BaseModel):
    """Represents the input payload a worker should handle."""

    source_uri: str = Field(..., description="Location of the input asset")
    job_type: str = Field(default="vision.process", description="Processing routine")
    parameters: dict[str, Any] = Field(default_factory=dict)


class JobCreateRequest(BaseModel):
    """Job submission request."""

    payload: JobPayload


class JobCreateResponse(BaseModel):
    """Response returned after a job is enqueued."""

    job_id: UUID = Field(default_factory=uuid4)
    status: Literal["queued", "received", "processing", "completed", "failed"] = (
        "queued"
    )
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    profile: Literal["cpu", "gpu", "rocm"] = "cpu"


class JobStatus(BaseModel):
    """Describes the runtime state of a job."""

    job_id: UUID
    status: Literal["queued", "received", "processing", "completed", "failed"]
    submitted_at: datetime
    updated_at: datetime
    message: str | None = None
    artifact_path: str | None = None
    profile: Literal["cpu", "gpu", "rocm"] = "cpu"


class HealthResponse(BaseModel):
    """Simple health status payload."""

    status: Literal["ok"] = "ok"
    service: str
    time: datetime = Field(default_factory=datetime.utcnow)
