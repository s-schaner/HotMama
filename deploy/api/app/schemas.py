"""Pydantic models shared by API components."""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover - type-checking only imports
    from .parsing import JobParser


_TIME_PATTERN = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})(?:\.(?P<millis>\d{1,3}))?$"
)


def _parse_timecode(value: str) -> float:
    match = _TIME_PATTERN.fullmatch(value)
    if not match:
        raise ValueError("timecode must match HH:MM:SS(.mmm)")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("minutes and seconds must be less than 60")
    millis_group = match.group("millis")
    millis = int(millis_group) if millis_group else 0
    if millis_group and len(millis_group) < 3:
        millis *= 10 ** (3 - len(millis_group))
    total_millis = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
    return total_millis / 1000.0


def _format_timecode(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    hours, remainder = divmod(total_millis, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    if millis:
        millis_str = f"{millis:03d}".rstrip("0")
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis_str}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Clip(BaseModel):
    """Represents a bounded time interval within an asset."""

    start: str = Field(..., description="Clip start time (HH:MM:SS[.mmm])")
    end: str = Field(..., description="Clip end time (HH:MM:SS[.mmm])")

    @field_validator("start", "end")
    @classmethod
    def normalize_timecode(cls, value: str) -> str:
        seconds = _parse_timecode(value.strip())
        return _format_timecode(seconds)

    @model_validator(mode="after")
    def ensure_order(self) -> "Clip":
        if _parse_timecode(self.end) <= _parse_timecode(self.start):
            raise ValueError("clip end must be greater than start")
        return self

    model_config = {"extra": "forbid"}


class JobPayload(BaseModel):
    """Represents the input payload a worker should handle."""

    task: Literal[
        "analyze_video",
        "extract_clips",
        "generate_heatmap",
        "detect_events",
    ] = Field(default="analyze_video", description="Requested operation")
    source_uri: str = Field(..., description="Location of the input asset")
    clips: list[Clip] = Field(default_factory=list, description="Optional clip ranges")
    fps: int = Field(default=30, ge=1, le=240, description="Frames per second override")
    model: str = Field(default="qwen-vision-default", description="Vision model name")
    options: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options", "parameters"),
        serialization_alias="options",
        description="Additional task-specific configuration",
    )

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("source_uri must not be empty")
        return trimmed

    model_config = {"extra": "forbid"}


class JobSpec(BaseModel):
    """Canonical job description consumed by workers."""

    payload: JobPayload
    priority: Literal["low", "normal", "high"] = Field(
        default="normal", description="Relative scheduling priority"
    )
    idempotency_key: str | None = Field(
        default=None, description="Client-supplied identifier used for dedupe"
    )

    model_config = {"extra": "forbid"}


class JobCreateRequest(BaseModel):
    """Job submission request supporting natural language input."""

    payload: JobPayload | None = None
    priority: Literal["low", "normal", "high"] | None = None
    idempotency_key: str | None = None
    nl: str | None = Field(
        default=None,
        description="Natural-language job request to be parsed by the API",
    )

    @field_validator("nl")
    @classmethod
    def normalize_nl(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def ensure_submission_mode(self) -> "JobCreateRequest":
        if self.payload is not None and self.nl is not None:
            raise ValueError("provide either 'payload' or 'nl', not both")
        if self.payload is None and self.nl is None:
            raise ValueError("payload or nl must be provided")
        return self

    def to_spec(self, parser: "JobParser" | None = None) -> JobSpec:
        if self.nl is not None:
            if parser is None:
                raise ValueError("parser is required when handling natural language")
            spec = parser.parse(self.nl)
        else:
            assert self.payload is not None
            spec = JobSpec(payload=self.payload)

        updates: dict[str, Any] = {}
        if self.priority is not None:
            updates["priority"] = self.priority
        if self.idempotency_key is not None:
            updates["idempotency_key"] = self.idempotency_key
        if updates:
            spec = spec.model_copy(update=updates)
        return spec

    model_config = {"extra": "forbid"}


class JobCreateResponse(BaseModel):
    """Response returned after a job is enqueued."""

    job_id: UUID = Field(default_factory=uuid4)
    status: Literal["queued", "received", "processing", "completed", "failed"] = (
        "queued"
    )
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    profile: Literal["cpu", "gpu", "rocm"] = "cpu"
    priority: Literal["low", "normal", "high"] = "normal"
    idempotency_key: str | None = None


class JobStatus(BaseModel):
    """Describes the runtime state of a job."""

    job_id: UUID
    status: Literal["queued", "received", "processing", "completed", "failed"]
    submitted_at: datetime
    updated_at: datetime
    message: str | None = None
    artifact_path: str | None = None
    profile: Literal["cpu", "gpu", "rocm"] = "cpu"
    priority: Literal["low", "normal", "high"] = "normal"


class HealthResponse(BaseModel):
    """Simple health status payload."""

    status: Literal["ok"] = "ok"
    service: str
    time: datetime = Field(default_factory=datetime.utcnow)
