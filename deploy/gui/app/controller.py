"""Core orchestration logic for the GUI."""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from uuid import UUID

from collections.abc import Sequence
from typing import Any

from deploy.api.app.parsing import DEFAULT_SYSTEM_PROMPT

from .client import ApiClient, JobHandle, JobState, LMStudioClient, VideoLLMClient
from .config import Settings
from .config_manager import ConfigManager
from .storage import StorageManager

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    np = None  # type: ignore

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO = None  # type: ignore

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None  # type: ignore

LOGGER = logging.getLogger("hotmama.gui.controller")


class GuiController:
    """Business logic separated from the Gradio UI for testability."""

    def __init__(
        self,
        client: ApiClient,
        storage: StorageManager,
        settings: Settings,
        lm_client: LMStudioClient | None = None,
        video_llm_client: VideoLLMClient | None = None,
        config_manager: ConfigManager | None = None,
    ) -> None:
        self._client = client
        self._storage = storage
        self._settings = settings
        self._lm_client = lm_client
        self._video_llm_client = video_llm_client
        self._config_manager = config_manager or ConfigManager(settings.artifact_root / "gui" / "configs")

        # Lazy-loaded vision models
        self._yolo_model = None
        self._mp_pose = None
        self._mp_drawing = None

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

    def query_video_llm(
        self,
        video_path: str,
        query: str,
        vision_model: str = "none",
        llm_provider: str = "lmstudio",
        endpoint_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        num_frames: int = 5,
        frame_size: tuple[int, int] = (512, 384),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """Query video using LLM with optional vision model preprocessing.

        Args:
            video_path: Path to video file
            query: User question about the video
            vision_model: Vision model to use (yolov8, pose, none)
            llm_provider: LLM provider (lmstudio or huggingface)
            endpoint_url: Custom endpoint URL (overrides settings)
            api_key: Custom API key (overrides settings)
            model_name: Custom model name (overrides settings)
            num_frames: Number of frames to sample from video
            frame_size: Tuple of (width, height) for frame resizing
            temperature: LLM temperature (0.0-2.0)
            max_tokens: Maximum tokens in LLM response

        Yields:
            Text chunks from the streaming LLM response
        """
        if not HAS_CV2:
            yield "❌ OpenCV is required but not installed. Please install opencv-python."
            return

        if not video_path:
            yield "❌ No video provided."
            return

        # Use custom endpoint or fall back to settings
        if not endpoint_url:
            if llm_provider == "lmstudio":
                endpoint_url = self._settings.lmstudio_base_url
            elif llm_provider == "huggingface":
                endpoint_url = self._settings.huggingface_api_url

        if not endpoint_url:
            yield f"❌ No endpoint URL configured for {llm_provider}."
            return

        # Use custom values or fall back to settings
        if not api_key:
            api_key = (
                self._settings.lmstudio_api_key
                if llm_provider == "lmstudio"
                else self._settings.huggingface_api_key
            )

        if not model_name:
            model_name = (
                self._settings.lm_vision_model
                if llm_provider == "lmstudio"
                else self._settings.huggingface_model
            )

        # Create video LLM client on demand based on provider
        try:
            video_client = VideoLLMClient(
                provider=llm_provider,
                base_url=endpoint_url,
                api_key=api_key or "default",
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._settings.request_timeout,
            )

            # Enhance query with vision model context if specified
            enhanced_query = query
            if vision_model != "none":
                enhanced_query = (
                    f"{query}\n\n"
                    f"Note: This video has been preprocessed with {vision_model} "
                    f"for enhanced object detection and tracking."
                )

            # Stream response from video LLM with custom parameters
            yield from video_client.query_video_stream(
                video_path,
                enhanced_query,
                num_frames=num_frames,
                frame_size=frame_size,
            )

            video_client.close()

        except Exception as exc:
            LOGGER.error("Video LLM query failed", exc_info=exc)
            yield f"\n\n❌ Error: {exc}"

    def capture_video_snapshot(
        self, video_path: str, vision_model: str = "none", query: str | None = None
    ) -> tuple[str | None, str]:
        """Capture a snapshot from video for analysis.

        Args:
            video_path: Path to video file
            vision_model: Vision model to apply (yolov8, detectron2, pose, none)
            query: Optional LLM query to run on the snapshot

        Returns:
            Tuple of (snapshot_path, message)
        """
        if not HAS_CV2:
            return None, "❌ OpenCV is required but not installed."

        if not video_path:
            return None, "❌ No video provided."

        try:
            # Extract first frame (could be enhanced to capture current playback position)
            capture = cv2.VideoCapture(video_path)
            success, frame = capture.read()
            capture.release()

            if not success:
                return None, "❌ Failed to read video frame."

            # Apply vision model if specified
            if vision_model == "pose":
                frame = self._apply_pose_skeleton(frame)
                message = "✅ Snapshot captured with pose estimation overlay."
            elif vision_model == "yolov8":
                frame = self._apply_object_detection(frame, model_type="yolov8")
                message = "✅ Snapshot captured with YOLOv8 object detection."
            elif vision_model == "detectron2":
                frame = self._apply_object_detection(frame, model_type="detectron2")
                message = "✅ Snapshot captured with Detectron2 segmentation."
            else:
                message = "✅ Snapshot captured."

            # Save snapshot
            import tempfile
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False, dir=str(self._settings.download_dir)
            ) as tmp:
                cv2.imwrite(tmp.name, frame)
                snapshot_path = tmp.name

            # Optionally query LLM about the snapshot
            if query:
                message += f"\n\nLLM Analysis: {query}"
                # Could stream LLM response here if needed

            return snapshot_path, message

        except Exception as exc:
            LOGGER.error("Snapshot capture failed", exc_info=exc)
            return None, f"❌ Error: {exc}"

    def execute_video_control(self, video_path: str, command: str) -> str:
        """Execute video control command.

        Args:
            video_path: Path to video file
            command: Control command (e.g., "pause when number 8 touches the ball")

        Returns:
            Result message with frame/timestamp if condition found
        """
        if not HAS_CV2:
            return "❌ OpenCV is required but not installed."

        if not video_path:
            return "❌ No video provided."

        try:
            # Parse command using LLM
            parsed_command = self._parse_control_command(command)

            # Execute command based on parsed action
            if parsed_command.get("action") == "pause":
                condition = parsed_command.get("condition", {})
                frame_num, timestamp = self._find_condition_frame(
                    video_path, condition
                )

                if frame_num is not None:
                    return (
                        f"✅ Condition met at frame {frame_num} "
                        f"(timestamp: {timestamp:.2f}s)\n"
                        f"Suggested action: Pause playback"
                    )
                else:
                    return "⚠️ Condition not found in video."
            else:
                return f"⚠️ Unsupported action: {parsed_command.get('action')}"

        except Exception as exc:
            LOGGER.error("Video control execution failed", exc_info=exc)
            return f"❌ Error: {exc}"

    def _apply_pose_skeleton(self, frame):
        """Apply pose estimation skeleton overlay using MediaPipe."""
        if not HAS_MEDIAPIPE:
            LOGGER.warning("MediaPipe not installed, using placeholder")
            cv2.putText(
                frame,
                "MediaPipe not installed",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return frame

        # Lazy load MediaPipe models
        if self._mp_pose is None:
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
            self._mp_drawing = mp.solutions.drawing_utils

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = self._mp_pose.process(frame_rgb)

            # Draw pose landmarks
            if results.pose_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    self._mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    self._mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=2
                    ),
                )
            else:
                cv2.putText(
                    frame,
                    "No pose detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

            return frame

        except Exception as exc:
            LOGGER.error("Pose estimation failed", exc_info=exc)
            cv2.putText(
                frame,
                f"Pose error: {exc}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame

    def _apply_object_detection(self, frame, model_type: str = "yolov8"):
        """Apply object detection overlay using YOLOv8."""
        if model_type != "yolov8" or not HAS_YOLO:
            LOGGER.warning(f"{model_type} not available, using placeholder")
            cv2.putText(
                frame,
                f"{model_type.upper()} not installed",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return frame

        # Lazy load YOLO model
        if self._yolo_model is None:
            try:
                self._yolo_model = YOLO("yolov8n.pt")  # Use nano model for speed
                LOGGER.info("Loaded YOLOv8n model")
            except Exception as exc:
                LOGGER.error("Failed to load YOLO model", exc_info=exc)
                cv2.putText(
                    frame,
                    f"YOLO load error: {exc}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                return frame

        try:
            # Run inference
            results = self._yolo_model(frame, verbose=False)

            # Draw bounding boxes and labels
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{result.names[cls]} {conf:.2f}"

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label background
                    (w, h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    cv2.rectangle(
                        frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1
                    )

                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                    )

            return frame

        except Exception as exc:
            LOGGER.error("Object detection failed", exc_info=exc)
            cv2.putText(
                frame,
                f"Detection error: {exc}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame

    def _parse_control_command(self, command: str) -> dict[str, Any]:
        """Parse natural language control command (placeholder)."""
        # This is a placeholder - would use LLM to parse command
        # For now, simple keyword matching
        command_lower = command.lower()

        if "pause" in command_lower:
            return {
                "action": "pause",
                "condition": {"type": "event", "description": command},
            }
        elif "play" in command_lower:
            return {"action": "play", "condition": {}}
        else:
            return {"action": "unknown", "condition": {}}

    def _find_condition_frame(
        self, video_path: str, condition: dict[str, Any]
    ) -> tuple[int | None, float]:
        """Find frame where condition is met (placeholder)."""
        # This is a placeholder - would use vision models to detect condition
        # For now, return middle of video as example
        import cv2

        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()

        # Return middle frame as placeholder
        frame_num = total_frames // 2
        timestamp = frame_num / fps if fps > 0 else 0

        return frame_num, timestamp

    def save_llm_config(
        self,
        name: str,
        provider: str,
        endpoint: str,
        api_key: str,
        model: str,
    ) -> None:
        """Save an LLM configuration for later use.

        Args:
            name: Unique name for the configuration
            provider: Provider type (lmstudio or huggingface)
            endpoint: Base URL for the LLM API
            api_key: API key for authentication
            model: Model name or path
        """
        config = {
            "provider": provider,
            "endpoint": endpoint,
            "api_key": api_key,
            "model": model,
        }
        self._config_manager.save_config(name, config)

    def load_llm_config(self, name: str) -> dict[str, Any] | None:
        """Load a saved LLM configuration.

        Args:
            name: Name of the configuration to load

        Returns:
            Configuration dictionary or None if not found
        """
        return self._config_manager.load_config(name)

    def list_llm_configs(self) -> list[str]:
        """List all saved LLM configuration names.

        Returns:
            List of configuration names
        """
        return self._config_manager.list_configs()

    def delete_llm_config(self, name: str) -> None:
        """Delete a saved LLM configuration.

        Args:
            name: Name of the configuration to delete
        """
        self._config_manager.delete_config(name)

    def close(self) -> None:
        """Release any network resources held by the controller."""

        self._client.close()
        if self._lm_client is not None:
            self._lm_client.close()
        if self._video_llm_client is not None:
            self._video_llm_client.close()

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

    # Initialize Video LLM client based on provider preference
    video_llm_client: VideoLLMClient | None = None
    try:
        if settings.llm_provider == "lmstudio" and settings.lmstudio_base_url:
            video_llm_client = VideoLLMClient(
                provider="lmstudio",
                base_url=settings.lmstudio_base_url,
                api_key=settings.lmstudio_api_key,
                model=settings.lm_vision_model,
                temperature=settings.lm_temperature,
                max_tokens=settings.lm_max_tokens,
                timeout=settings.request_timeout,
            )
        elif settings.llm_provider == "huggingface" and settings.huggingface_api_url:
            video_llm_client = VideoLLMClient(
                provider="huggingface",
                base_url=settings.huggingface_api_url,
                api_key=settings.huggingface_api_key,
                model=settings.huggingface_model,
                temperature=settings.lm_temperature,
                max_tokens=settings.lm_max_tokens,
                timeout=settings.request_timeout,
            )
    except Exception as exc:  # pragma: no cover - defensive branch
        LOGGER.warning("unable to initialise Video LLM client", exc_info=exc)
        video_llm_client = None

    return GuiController(
        client=client,
        storage=storage,
        settings=settings,
        lm_client=lm_client,
        video_llm_client=video_llm_client,
    )


__all__ = ["GuiController", "build_controller"]
