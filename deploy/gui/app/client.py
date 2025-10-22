"""HTTP client used by the GUI to communicate with the API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx


@dataclass(slots=True)
class JobHandle:
    """Represents the initial response after submitting a job."""

    job_id: UUID
    status: str
    submitted_at: datetime
    profile: str


@dataclass(slots=True)
class JobState:
    """Represents the current state of a job."""

    job_id: UUID
    status: str
    submitted_at: datetime
    updated_at: datetime
    profile: str
    message: str | None
    artifact_path: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable representation for UI rendering."""

        return {
            "job_id": str(self.job_id),
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "profile": self.profile,
            "message": self.message or "",
            "artifact_path": self.artifact_path or "",
        }


class ApiClient:
    """Thin wrapper around the API HTTP endpoints."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout
        )

    def submit_job(
        self,
        source_uri: str,
        parameters: dict[str, Any] | None = None,
        job_type: str = "vision.process",
    ) -> JobHandle:
        payload: dict[str, Any] = {
            "payload": {
                "source_uri": source_uri,
                "job_type": job_type,
                "parameters": parameters or {},
            }
        }
        response = self._client.post("/jobs", json=payload)
        self._raise_for_status(response)
        data = response.json()
        return JobHandle(
            job_id=UUID(data["job_id"]),
            status=data.get("status", "queued"),
            submitted_at=datetime.fromisoformat(data["submitted_at"]),
            profile=data.get("profile", "cpu"),
        )

    def get_status(self, job_id: UUID) -> JobState:
        response = self._client.get(f"/jobs/{job_id}")
        self._raise_for_status(response)
        data = response.json()
        return JobState(
            job_id=UUID(data["job_id"]),
            status=data["status"],
            submitted_at=datetime.fromisoformat(data["submitted_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            profile=data.get("profile", "cpu"),
            message=data.get("message"),
            artifact_path=data.get("artifact_path"),
        )

    def download_artifact(self, job_id: UUID, destination: Path) -> Path:
        response = self._client.get(f"/jobs/{job_id}/artifact")
        self._raise_for_status(response)
        filename = self._resolve_filename(response.headers.get("content-disposition"))
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / filename
        target.write_bytes(response.content)
        return target

    def close(self) -> None:
        self._client.close()

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - thin wrapper
            detail: str
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:  # pragma: no cover - fallback for non-json errors
                detail = str(exc)
            raise RuntimeError(detail) from exc

    def _resolve_filename(self, content_disposition: str | None) -> str:
        if not content_disposition:
            return "artifact.bin"
        for part in content_disposition.split(";"):
            part = part.strip()
            if part.startswith("filename="):
                return part.split("=", 1)[1].strip('"') or "artifact.bin"
        return "artifact.bin"

    def __enter__(self) -> "ApiClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, *_: Any) -> None:  # pragma: no cover - convenience
        self.close()


__all__ = ["ApiClient", "JobHandle", "JobState"]
