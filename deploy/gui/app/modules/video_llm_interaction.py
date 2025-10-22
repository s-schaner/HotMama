"""Video LLM Interaction module for real-time video Q&A and analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr

    from deploy.gui.app.controller import GuiController

from .base import GuiModule

LOGGER = logging.getLogger("hotmama.gui.modules.video_llm")


class VideoLLMInteractionModule(GuiModule):
    """Module for interactive video analysis with LLM."""

    @property
    def name(self) -> str:
        return "video_llm_interaction"

    @property
    def display_name(self) -> str:
        return "Video LLM Interaction"

    def build_ui(self, controller: GuiController) -> gr.Blocks:
        """Build the Video LLM Interaction UI."""
        import gradio as gr

        # Helper functions for event handlers
        def handle_query_submit(
            video_path: str | None,
            query: str,
            vision_model: str,
            llm_provider: str,
        ):
            """Handle query submission and stream response."""
            if not video_path:
                yield "⚠️ Please upload a video first."
                return

            if not query.strip():
                yield "⚠️ Please enter a query."
                return

            yield "Processing your query...\n\n"

            try:
                # Stream response from controller
                for chunk in controller.query_video_llm(
                    video_path=video_path,
                    query=query.strip(),
                    vision_model=vision_model,
                    llm_provider=llm_provider,
                ):
                    yield chunk
            except Exception as exc:
                LOGGER.error("Video LLM query failed", exc_info=exc)
                yield f"\n\n❌ Error: {exc}"

        def handle_snapshot_capture(
            video_path: str | None, vision_model: str, llm_query: str
        ):
            """Capture current video frame for analysis."""
            if not video_path:
                return None, "⚠️ Please upload a video first."

            try:
                snapshot_path, message = controller.capture_video_snapshot(
                    video_path=video_path,
                    vision_model=vision_model,
                    query=llm_query.strip() if llm_query else None,
                )
                return snapshot_path, message
            except Exception as exc:
                LOGGER.error("Snapshot capture failed", exc_info=exc)
                return None, f"❌ Error: {exc}"

        def handle_video_control(video_path: str | None, control_command: str):
            """Execute video control command."""
            if not video_path:
                return "⚠️ Please upload a video first."

            if not control_command.strip():
                return "⚠️ Please enter a control command."

            try:
                result = controller.execute_video_control(
                    video_path=video_path, command=control_command.strip()
                )
                return result
            except Exception as exc:
                LOGGER.error("Video control failed", exc_info=exc)
                return f"❌ Error: {exc}"

        def handle_video_upload(video_path: str | None):
            """Handle video upload and display preview."""
            if not video_path:
                return None
            return video_path

        # Build the UI
        with gr.Blocks() as module_ui:
            gr.Markdown("### Video LLM Interaction")
            gr.Markdown(
                "Upload a video clip and ask questions about its content. "
                "The LLM will analyze the video and provide answers in real-time."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    # Video player section
                    video_player = gr.Video(
                        label="Video Clip",
                        interactive=True,
                        height=400,
                        format=None,
                    )

                    # Vision model selector
                    vision_model_selector = gr.Dropdown(
                        label="Vision Model",
                        choices=[
                            ("YOLOv8 - Object Detection", "yolov8"),
                            ("Detectron2 - Instance Segmentation", "detectron2"),
                            ("Pose Estimation - Skeletal Analysis", "pose"),
                            ("None - LLM Only", "none"),
                        ],
                        value="yolov8",
                        info="Select the computer vision model for video analysis",
                    )

                    # LLM provider selector
                    llm_provider_selector = gr.Radio(
                        label="LLM Provider",
                        choices=[
                            ("LM Studio (Local)", "lmstudio"),
                            ("Hugging Face", "huggingface"),
                        ],
                        value="lmstudio",
                        info="Choose the LLM inference provider",
                    )

                with gr.Column(scale=2):
                    # Query section
                    query_box = gr.Textbox(
                        label="Query",
                        placeholder="Ask a question about the video (e.g., 'Who is the last player that touched the ball?')",
                        lines=3,
                    )

                    # Reply section with streaming
                    reply_box = gr.Textbox(
                        label="Reply",
                        placeholder="Response will stream here...",
                        lines=10,
                        interactive=False,
                    )

                    # Action buttons
                    with gr.Row():
                        submit_query_btn = gr.Button(
                            "Send Query", variant="primary", icon="🔍"
                        )
                        clear_reply_btn = gr.Button(
                            "Clear Reply", variant="secondary", icon="🧹"
                        )

            # Snapshot and control section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Video Snapshot Analysis")
                    gr.Markdown(
                        "Capture a frame from the video for detailed analysis with both CV models and LLM."
                    )

                    snapshot_btn = gr.Button(
                        "Capture Snapshot", variant="primary", icon="📸"
                    )

                    snapshot_output = gr.Image(
                        label="Captured Frame", interactive=False, height=300
                    )

                    snapshot_feedback = gr.Textbox(
                        label="Snapshot Analysis",
                        placeholder="Analysis results will appear here...",
                        lines=5,
                        interactive=False,
                    )

                with gr.Column():
                    gr.Markdown("#### Video Control Commands")
                    gr.Markdown(
                        "Execute intelligent video control (e.g., 'pause when number 8 touches the ball')."
                    )

                    control_command_box = gr.Textbox(
                        label="Control Command",
                        placeholder="Enter control command (e.g., 'pause when number 8 touches the ball')",
                        lines=2,
                    )

                    execute_control_btn = gr.Button(
                        "Execute Control", variant="primary", icon="⚡"
                    )

                    control_feedback = gr.Textbox(
                        label="Control Result",
                        placeholder="Control execution results...",
                        lines=5,
                        interactive=False,
                    )

            # Wire up event handlers
            video_player.upload(
                handle_video_upload, inputs=[video_player], outputs=[video_player]
            )

            submit_query_btn.click(
                handle_query_submit,
                inputs=[
                    video_player,
                    query_box,
                    vision_model_selector,
                    llm_provider_selector,
                ],
                outputs=[reply_box],
            )

            clear_reply_btn.click(lambda: "", outputs=[reply_box])

            snapshot_btn.click(
                handle_snapshot_capture,
                inputs=[video_player, vision_model_selector, query_box],
                outputs=[snapshot_output, snapshot_feedback],
            )

            execute_control_btn.click(
                handle_video_control,
                inputs=[video_player, control_command_box],
                outputs=[control_feedback],
            )

        return module_ui


__all__ = ["VideoLLMInteractionModule"]
