"""Core orchestration logic for the GUI."""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from uuid import UUID

from .client import ApiClient, JobHandle, JobState
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
    ) -> None:
        self._client = client
        self._storage = storage
        self._settings = settings

    def submit_job(
        self,
        upload_path: str | None,
        parameters_json: str,
        job_type: str = "vision.process",
    ) -> tuple[str, str]:
        if not upload_path:
            return "", "⚠️ Please upload a video before submitting."

        try:
            parameters = self._parse_parameters(parameters_json)
        except JSONDecodeError as exc:
            return "", f"❌ Invalid parameters JSON: {exc}"

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
            handle = self._client.submit_job(
                str(stored_path), parameters=parameters, job_type=job_type
            )
        except RuntimeError as exc:
            LOGGER.error("job submission failed", exc_info=exc)
            return "", f"❌ Failed to submit job: {exc}"

        message = self._format_submission_message(handle, job_type)
        return str(handle.job_id), message

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

    def _parse_parameters(self, payload: str) -> dict:
        if not payload or not payload.strip():
            return {}
        return json.loads(payload)

    def _format_submission_message(self, handle: JobHandle, job_type: str) -> str:
        return (
            "✅ Job {job} queued on {profile} profile at {time}".format(
                job=handle.job_id,
                profile=handle.profile.upper(),
                time=handle.submitted_at.isoformat(timespec="seconds"),
            )
            + f"\n• Task: {job_type}"
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


def build_controller(settings: Settings) -> GuiController:
    """Factory used by the runtime entrypoint."""

    client = ApiClient(str(settings.api_base_url), timeout=settings.request_timeout)
    storage = StorageManager(settings.upload_dir, settings.download_dir)
    return GuiController(client=client, storage=storage, settings=settings)


__all__ = ["GuiController", "build_controller"]
