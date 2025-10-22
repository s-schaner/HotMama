"""Core orchestration logic for the GUI."""

from __future__ import annotations

import json
import logging
import re
from json import JSONDecodeError
from uuid import UUID

from .client import ApiClient, JobHandle, JobState
from .config import Settings
from .storage import StorageManager

LOGGER = logging.getLogger("hotmama.gui.controller")

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
        clip_ranges: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        if not upload_path:
            return "", "⚠️ Please upload a video before submitting."

        try:
            parameters = self._parse_parameters(parameters_json)
        except JSONDecodeError as exc:
            return "", f"❌ Invalid parameters JSON: {exc}"

        try:
            clips = self._parse_clips(clip_ranges)
        except ValueError as exc:
            return "", f"❌ Invalid clip configuration: {exc}"

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
                str(stored_path),
                parameters=parameters,
                job_type=job_type,
                clips=clips or None,
            )
        except RuntimeError as exc:
            LOGGER.error("job submission failed", exc_info=exc)
            return "", f"❌ Failed to submit job: {exc}"

        message = self._format_submission_message(handle)
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

    def _parse_clips(
        self, clip_ranges: list[dict[str, str]] | None
    ) -> list[dict[str, str]]:
        if not clip_ranges:
            return []
        if not isinstance(clip_ranges, list):
            raise ValueError("clips must be provided as a list")

        normalised: list[dict[str, str]] = []
        for index, entry in enumerate(clip_ranges, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"clip {index} is not a mapping")
            start_value = entry.get("start")
            end_value = entry.get("end")
            if not isinstance(start_value, str) or not isinstance(end_value, str):
                raise ValueError(
                    f"clip {index} requires start and end time strings"
                )
            start_text = start_value.strip()
            end_text = end_value.strip()
            if not start_text or not end_text:
                raise ValueError(
                    f"clip {index} requires both start and end times"
                )
            try:
                start_seconds = _parse_timecode(start_text)
                end_seconds = _parse_timecode(end_text)
            except ValueError as exc:
                raise ValueError(f"clip {index}: {exc}") from exc
            if end_seconds <= start_seconds:
                raise ValueError(
                    f"clip {index}: end must be greater than start"
                )
            normalised.append(
                {
                    "start": _format_timecode(start_seconds),
                    "end": _format_timecode(end_seconds),
                }
            )
        return normalised

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


def build_controller(settings: Settings) -> GuiController:
    """Factory used by the runtime entrypoint."""

    client = ApiClient(str(settings.api_base_url), timeout=settings.request_timeout)
    storage = StorageManager(settings.upload_dir, settings.download_dir)
    return GuiController(client=client, storage=storage, settings=settings)


__all__ = ["GuiController", "build_controller"]
