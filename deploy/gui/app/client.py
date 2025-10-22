"""HTTP client used by the GUI to communicate with the API."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from deploy.api.app.parsing import DEFAULT_SYSTEM_PROMPT
from deploy.api.app.schemas import JobSpec

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

LOGGER = logging.getLogger("hotmama.gui.client")

_VALID_TASKS = {
    "analyze_video",
    "extract_clips",
    "generate_heatmap",
    "detect_events",
}

_TASK_DEFAULT_OPTIONS: dict[str, dict[str, Any]] = {
    "generate_heatmap": {"overlays": ["heatmap"]},
}

_TASK_ALIASES = {
    "pipeline.analysis": "analyze_video",
    "pipeline.segment": "extract_clips",
    "pipeline.events": "detect_events",
    "vision.process": "analyze_video",
    "vision.segment": "extract_clips",
    "vision.heatmap": "generate_heatmap",
    "vision.events": "detect_events",
}

_OVERLAY_ALIASES = {
    "overlay.activity_heatmap": "heatmap",
    "overlay.object_tracking": "tracking",
    "overlay.pose_skeleton": "pose",
}

_KNOWN_OVERLAYS = {"heatmap", "tracking", "pose"}


@dataclass(slots=True)
class JobHandle:
    """Represents the initial response after submitting a job."""

    job_id: UUID
    status: str
    submitted_at: datetime
    profile: str
    priority: str
    idempotency_key: str | None
    task: str


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
    priority: str

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
            "priority": self.priority,
        }


class LMStudioClient:
    """Minimal helper for interacting with an LM Studio deployment."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        system_prompt: str | None = None,
        instruct_model: str = "qwen2.5-3b-instruct",
        enrichment_model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("LM Studio base URL is required")
        if not instruct_model:
            raise ValueError("LM Studio model name is required")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._instruct_model = instruct_model
        self._enrichment_model = enrichment_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._schema = JobSpec.model_json_schema()

    def generate_manifest(
        self,
        prompt: str,
        *,
        source_uri: str,
        task_hint: str | None = None,
        enrich: bool = False,
    ) -> JobSpec:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        manifest = self._request_manifest(
            prompt=prompt.strip(), source_uri=source_uri, task_hint=task_hint
        )
        if enrich and self._enrichment_model:
            try:
                manifest = self._enrich_manifest(
                    prompt=prompt.strip(), manifest=manifest
                )
            except RuntimeError as exc:
                LOGGER.warning("manifest enrichment failed", exc_info=exc)
        return manifest

    def close(self) -> None:
        self._client.close()

    def _request_manifest(
        self,
        *,
        prompt: str,
        source_uri: str,
        task_hint: str | None,
    ) -> JobSpec:
        user_prompt = prompt
        hints: list[str] = []
        if source_uri:
            hints.append(f"Video path: {source_uri}")
        if task_hint:
            hints.append(f"Preferred task: {task_hint}")
        if hints:
            user_prompt = f"{user_prompt}\n\n" + "\n".join(hints)
        payload = self._post_chat_completion(
            model=self._instruct_model,
            user_content=user_prompt,
            temperature=self._temperature,
        )
        return self._extract_spec(payload)

    def _enrich_manifest(self, *, prompt: str, manifest: JobSpec) -> JobSpec:
        if not self._enrichment_model:
            return manifest
        manifest_json = json.dumps(
            manifest.model_dump(mode="json"), indent=2, sort_keys=True
        )
        user_content = (
            "Refine the existing manifest if additional structure is valuable.\n"
            "Always return a full JSON document matching the schema.\n\n"
            f"Operator prompt:\n{prompt}\n\n"
            f"Current manifest:\n{manifest_json}"
        )
        payload = self._post_chat_completion(
            model=self._enrichment_model,
            user_content=user_content,
            temperature=self._temperature,
        )
        return self._extract_spec(payload)

    def _post_chat_completion(
        self,
        *,
        model: str,
        user_content: str,
        temperature: float,
    ) -> dict[str, Any]:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "job_spec",
                    "strict": True,
                    "schema": self._schema,
                },
            },
        }
        try:
            response = self._client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            raise RuntimeError("LM Studio request failed") from exc
        return response.json()

    def _extract_spec(self, payload: dict[str, Any]) -> JobSpec:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LM Studio returned malformed response") from exc
        if not content:
            raise RuntimeError("LM Studio returned empty content")
        try:
            return JobSpec.model_validate_json(content)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError("LM Studio response failed validation") from exc


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
        job_type: str = "pipeline.analysis",
        overlays: Sequence[str] | None = None,
        priority: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> JobHandle:
        task = self._normalise_task(job_type)
        if manifest is not None:
            body = self._prepare_manifest_body(
                manifest,
                source_uri=source_uri,
                task=task,
                overlays=overlays,
                parameters=parameters,
                priority=priority,
            )
        else:
            options = self._build_options(
                task=task, parameters=parameters, overlays=overlays
            )
            body = {
                "payload": {
                    "task": task,
                    "source_uri": source_uri,
                    "options": options,
                }
            }
            if priority:
                body["priority"] = priority

        response = self._client.post("/jobs", json=body)
        self._raise_for_status(response)
        data = response.json()
        return JobHandle(
            job_id=UUID(data["job_id"]),
            status=data.get("status", "queued"),
            submitted_at=datetime.fromisoformat(data["submitted_at"]),
            profile=data.get("profile", "cpu"),
            priority=data.get("priority", body.get("priority", "normal")),
            idempotency_key=data.get("idempotency_key"),
            task=task,
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
            priority=data.get("priority", "normal"),
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

    @staticmethod
    def _normalise_task(task: str) -> str:
        normalised = _TASK_ALIASES.get(task, task or "")
        if normalised in _VALID_TASKS:
            return normalised
        return "analyze_video"

    @staticmethod
    def _normalise_overlay_modes(
        overlays: Sequence[str] | None,
    ) -> list[str] | None:
        if overlays is None:
            return None
        normalised: list[str] = []
        for item in overlays:
            key = _OVERLAY_ALIASES.get(item, item)
            key = str(key).strip().lower()
            if not key:
                continue
            if key in _KNOWN_OVERLAYS and key not in normalised:
                normalised.append(key)
        return normalised

    def _build_options(
        self,
        *,
        task: str,
        parameters: dict[str, Any] | None,
        overlays: Sequence[str] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        defaults = _TASK_DEFAULT_OPTIONS.get(task)
        if defaults:
            options.update(deepcopy(defaults))
        if parameters:
            options.update(deepcopy(parameters))
        overlay_list = self._normalise_overlay_modes(overlays)
        if overlay_list is not None:
            options["overlays"] = overlay_list
        return options

    def _prepare_manifest_body(
        self,
        manifest: dict[str, Any],
        *,
        source_uri: str,
        task: str,
        overlays: Sequence[str] | None,
        parameters: dict[str, Any] | None,
        priority: str | None,
    ) -> dict[str, Any]:
        body = deepcopy(manifest)
        payload = body.setdefault("payload", {})
        if not isinstance(payload, dict):
            raise RuntimeError("payload must be an object")
        payload.setdefault("task", task)
        payload["source_uri"] = source_uri
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise RuntimeError("payload.options must be an object")
        combined_options = deepcopy(options)
        if parameters:
            combined_options.update(deepcopy(parameters))
        overlay_list = self._normalise_overlay_modes(overlays)
        if overlay_list is not None:
            combined_options["overlays"] = overlay_list
        payload["options"] = combined_options
        if priority and "priority" not in body:
            body["priority"] = priority
        return body


class VideoLLMClient:
    """Client for streaming video LLM queries with vision support."""

    def __init__(
        self,
        provider: str = "lmstudio",
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._provider = provider.lower()
        self._base_url = base_url
        self._api_key = api_key or "default"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        if not base_url:
            raise ValueError(f"{provider} base URL is required")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def query_video_stream(
        self,
        video_path: str,
        query: str,
        *,
        frame_indices: list[int] | None = None,
        num_frames: int = 5,
        frame_size: tuple[int, int] = (512, 384),
    ):
        """Stream LLM response for video query.

        Args:
            video_path: Path to video file
            query: User question about the video
            frame_indices: Optional list of frame indices to analyze
            num_frames: Number of frames to sample (if frame_indices not provided)
            frame_size: Tuple of (width, height) for frame resizing

        Yields:
            Text chunks from the LLM response
        """
        if not HAS_CV2:
            yield "❌ OpenCV (cv2) is required for video processing but not installed."
            return

        # Extract frames from video
        frames_data = self._extract_frames(
            video_path, frame_indices, num_frames=num_frames, frame_size=frame_size
        )
        if not frames_data:
            yield "❌ Failed to extract frames from video."
            return

        # Query LLM with streaming
        try:
            if self._provider == "lmstudio":
                yield from self._query_lmstudio_stream(query, frames_data)
            elif self._provider == "huggingface":
                yield from self._query_huggingface_stream(query, frames_data)
            else:
                yield f"❌ Unsupported provider: {self._provider}"
        except Exception as exc:
            LOGGER.error("Video LLM query failed", exc_info=exc)
            yield f"\n\n❌ Query failed: {exc}"

    def _extract_frames(
        self,
        video_path: str,
        frame_indices: list[int] | None = None,
        num_frames: int = 5,
        frame_size: tuple[int, int] = (512, 384),
    ) -> list[str]:
        """Extract frames from video and encode as base64.

        Args:
            video_path: Path to video file
            frame_indices: Optional specific frame indices
            num_frames: Number of frames to sample uniformly (if frame_indices not provided)
            frame_size: Tuple of (width, height) for resizing

        Returns:
            List of base64-encoded JPEG frames
        """
        import base64

        try:
            capture = cv2.VideoCapture(video_path)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_indices is None:
                # Sample num_frames uniformly across the video
                frame_indices = [
                    int(i * total_frames / num_frames)
                    for i in range(num_frames)
                    if i * total_frames / num_frames < total_frames
                ]

            frames_b64 = []
            for frame_idx in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = capture.read()

                if success:
                    # Resize frame with custom size
                    frame_resized = cv2.resize(frame, frame_size)
                    # Encode as JPEG
                    _, buffer = cv2.imencode(".jpg", frame_resized)
                    frame_b64 = base64.b64encode(buffer).decode("utf-8")
                    frames_b64.append(frame_b64)

            capture.release()
            return frames_b64

        except Exception as exc:
            LOGGER.error("Frame extraction failed", exc_info=exc)
            return []

    def _query_lmstudio_stream(self, query: str, frames_b64: list[str]):
        """Query LM Studio with vision model using streaming."""
        # Build messages with images
        content = [{"type": "text", "text": query}]
        for frame_b64 in frames_b64[:3]:  # Limit to 3 frames to avoid token limits
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        payload = {
            "model": self._model or "qwen/qwen2.5-vl-7b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful video analysis assistant. Analyze the provided video frames and answer the user's question accurately.",
                },
                {"role": "user", "content": content},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }

        try:
            with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content_chunk = delta.get("content", "")
                            if content_chunk:
                                yield content_chunk
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPError as exc:
            raise RuntimeError(f"LM Studio request failed: {exc}") from exc

    def _query_huggingface_stream(self, query: str, frames_b64: list[str]):
        """Query Hugging Face Inference API with vision model."""
        # Note: Hugging Face API may vary by model; this is a generic implementation
        payload = {
            "inputs": {
                "question": query,
                "images": frames_b64[:3],  # Limit to 3 frames
            },
            "parameters": {
                "temperature": self._temperature,
                "max_new_tokens": self._max_tokens,
            },
            "stream": True,
        }

        try:
            with self._client.stream("POST", "/", json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        # Extract text from response (format may vary by model)
                        if isinstance(data, dict):
                            text = data.get("token", {}).get("text", "")
                            if text:
                                yield text
                        elif isinstance(data, list) and data:
                            text = data[0].get("generated_text", "")
                            if text:
                                yield text
                                return  # HF may return all at once

                    except json.JSONDecodeError:
                        continue

        except httpx.HTTPError as exc:
            raise RuntimeError(f"Hugging Face request failed: {exc}") from exc

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


__all__ = ["ApiClient", "JobHandle", "JobState", "LMStudioClient", "VideoLLMClient"]
