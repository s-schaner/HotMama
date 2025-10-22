"""Schemas for worker job processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class QueuedJob(BaseModel):
    job_id: UUID
    submitted_at: datetime
    status: Literal["queued", "received", "processing", "completed", "failed"]
    profile: Literal["cpu", "gpu", "rocm"]
    priority: Literal["low", "normal", "high"] = "normal"
    payload: dict[str, Any] = Field(default_factory=dict)


class ProcessResult(BaseModel):
    status: Literal["completed", "failed"]
    message: str
    artifact_path: str | None = None
