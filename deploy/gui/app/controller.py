"""Core orchestration logic for the GUI."""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from uuid import UUID

from collections.abc import Sequence
from typing import Any

from deploy.api.app.parsing import DEFAULT_SYSTEM_PROMPT

from .client import ApiClient, JobHandle, JobState, LMStudioClient
from .config import Settings
from .storage import StorageManager

LOGGER = logging.getLogger("hotmama.gui.controller")


class GuiController:
    """Business logic separated from the Gradio UI for testability."""

    def __init__(
        self,
        client: ApiClient,
        storage: StorageManager,
        settings: Settings,
        lm_client: LMStudioClient | None = None,
    ) -> None:
        self._client = client
        self._storage = storage
        self._settings = settings
        self._lm_client = lm_client

    def submit_job(
        self,
        upload_path: str | None,
        parameters_json: str,
        job_type: str = "pipeline.analysis",
        overlay_modes: Sequence[str] | None = None,
        input_mode: str = "json",
        prompt_text: str = "",
        enrich_manifest: bool = False,
    ) -> tuple[str, str, dict[str, Any] | None]:
        if not upload_path:
            return "", "⚠️ Please upload a video before submitting.", None

        stored_path = self._storage.save_upload(upload_path)
        LOGGER.info(
            "submitting job",
            extra={
                "path": str(stored_path),
                "job_type": job_type,
                "service": self._settings.service_name,
            },
        )

        try:
            manifest = self._compose_manifest(
                parameters_json,
                source_path=str(stored_path),
                job_type=job_type,
                overlay_modes=overlay_modes,
                input_mode=input_mode,
                prompt_text=prompt_text,
                enrich=enrich_manifest,
            )
        except JSONDecodeError as exc:
            return "", f"❌ Parameters JSON is invalid: {exc}", None
        except ValueError as exc:
            return "", f"❌ {exc}", None
        except RuntimeError as exc:
            LOGGER.error("manifest preparation failed", exc_info=exc)
            prefix = (
                "Failed to parse natural language request"
                if input_mode == "nl"
                else "Failed to prepare job manifest"
            )
            return "", f"❌ {prefix}: {exc}", None

        try:
            handle = self._client.submit_job(
                str(stored_path),
                job_type=job_type,
                overlays=list(overlay_modes) if overlay_modes is not None else None,
                manifest=manifest,
            )
        except RuntimeError as exc:
            LOGGER.error("job submission failed", exc_info=exc)
            return "", f"❌ Failed to submit job: {exc}", None

        message = self._format_submission_message(handle)
        return str(handle.job_id), message, manifest

    def refresh_status(self, job_id_text: str) -> tuple[dict[str, str], str, str]:
        if not job_id_text:
            return {}, "⚠️ Provide a job identifier to query status.", ""

        try:
            job_id = UUID(job_id_text)
        except ValueError:
            return {}, "❌ Invalid job identifier.", ""

        try:
            state = self._client.get_status(job_id)
        except RuntimeError as exc:
            LOGGER.error("status lookup failed", exc_info=exc)
            return {}, f"❌ Unable to fetch status: {exc}", ""

        artifact_path = ""
        if state.artifact_path:
            try:
                prepared = self._storage.prepare_artifact(state.artifact_path)
                artifact_path = str(prepared)
            except FileNotFoundError:
                artifact_path = ""

        message = state.message or self._format_status_message(state)
        return state.as_dict(), message, artifact_path

    def _format_submission_message(self, handle: JobHandle) -> str:
        return (
            "✅ Job {job} queued on {profile} profile at {time}".format(
                job=handle.job_id,
                profile=handle.profile.upper(),
                time=handle.submitted_at.isoformat(timespec="seconds"),
            )
            + f"\n• Task: {handle.task}"
            + f"\n• Priority: {handle.priority}"
        )

    def _format_status_message(self, state: JobState) -> str:
        base = f"Status: {state.status.upper()}"
        if state.status.lower() == "completed":
            base = f"✅ {base}"
        elif state.status.lower() == "failed":
            base = f"❌ {base}"
        return f"{base} (updated {state.updated_at.isoformat(timespec='seconds')})"

    def close(self) -> None:
        """Release any network resources held by the controller."""

        self._client.close()
        if self._lm_client is not None:
            self._lm_client.close()

    @property
    def supports_natural_language(self) -> bool:
        """Return True when an LM client is configured."""

        return self._lm_client is not None

    def _compose_manifest(
        self,
        payload_text: str,
        *,
        source_path: str,
        job_type: str,
        overlay_modes: Sequence[str] | None,
        input_mode: str,
        prompt_text: str,
        enrich: bool,
    ) -> dict[str, Any]:
        task = ApiClient._normalise_task(job_type)
        if input_mode == "nl":
            if self._lm_client is None:
                raise ValueError(
                    "Natural language mode is not available on this deployment."
                )
            manifest_model = self._lm_client.generate_manifest(
                prompt_text,
                source_uri=source_path,
                task_hint=task,
                enrich=enrich,
            )
            manifest = manifest_model.model_dump(mode="json")
        else:
            manifest = self._coerce_manifest(payload_text)

        payload = manifest.setdefault("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object.")
        payload.setdefault("task", task)
        payload["source_uri"] = source_path
        options = payload.get("options")
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("payload.options must be a JSON object.")
        overlay_list = ApiClient._normalise_overlay_modes(overlay_modes)
        if overlay_list is not None:
            merged = dict(options)
            merged["overlays"] = overlay_list
            options = merged
        payload["options"] = options
        return manifest

    def _coerce_manifest(self, payload_text: str) -> dict[str, Any]:
        if not payload_text or not payload_text.strip():
            return {"payload": {"options": {}}}
        data = json.loads(payload_text)
        if not isinstance(data, dict):
            raise ValueError("Parameters JSON must decode to an object.")
        if "payload" in data or "priority" in data or "idempotency_key" in data:
            payload = data.get("payload")
            if payload is None:
                data["payload"] = {"options": {}}
                return data
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object.")
            options = payload.get("options")
            if options is None:
                payload["options"] = {}
            elif not isinstance(options, dict):
                raise ValueError("payload.options must be a JSON object.")
            return data
        return {"payload": {"options": data}}


def build_controller(settings: Settings) -> GuiController:
    """Factory used by the runtime entrypoint."""

    client = ApiClient(str(settings.api_base_url), timeout=settings.request_timeout)
    storage = StorageManager(settings.upload_dir, settings.download_dir)
    lm_client: LMStudioClient | None = None
    if settings.lmstudio_base_url and settings.lm_parser_model:
        try:
            lm_client = LMStudioClient(
                settings.lmstudio_base_url,
                settings.lmstudio_api_key,
                system_prompt=settings.lm_system_prompt or DEFAULT_SYSTEM_PROMPT,
                instruct_model=settings.lm_parser_model,
                enrichment_model=settings.lm_enrichment_model,
                temperature=settings.lm_temperature,
                max_tokens=settings.lm_max_tokens,
                timeout=settings.request_timeout,
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("unable to initialise LM Studio client", exc_info=exc)
            lm_client = None
    return GuiController(
        client=client, storage=storage, settings=settings, lm_client=lm_client
    )


__all__ = ["GuiController", "build_controller"]
