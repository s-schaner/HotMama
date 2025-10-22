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
            endpoint_url: str,
            api_key: str,
            model_name: str,
            num_frames: int,
            frame_width: int,
            frame_height: int,
            temperature: float,
            max_tokens: int,
        ):
            """Handle query submission and stream response."""
            if not video_path:
                yield "⚠️ Please upload a video first."
                return

            if not query.strip():
                yield "⚠️ Please enter a query."
                return

            if not endpoint_url.strip():
                yield "⚠️ Please set the LLM endpoint URL."
                return

            yield "Processing your query...\n\n"

            try:
                # Stream response from controller with custom parameters
                for chunk in controller.query_video_llm(
                    video_path=video_path,
                    query=query.strip(),
                    vision_model=vision_model,
                    llm_provider=llm_provider,
                    endpoint_url=endpoint_url.strip(),
                    api_key=api_key.strip() if api_key else "default",
                    model_name=model_name.strip() if model_name else None,
                    num_frames=num_frames,
                    frame_size=(frame_width, frame_height),
                    temperature=temperature,
                    max_tokens=max_tokens,
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

        def handle_save_config(
            config_name: str,
            endpoint_url: str,
            api_key: str,
            model_name: str,
            llm_provider: str,
        ):
            """Save current configuration."""
            if not config_name.strip():
                return "⚠️ Please enter a configuration name.", gr.update()

            try:
                controller.save_llm_config(
                    name=config_name.strip(),
                    provider=llm_provider,
                    endpoint=endpoint_url.strip(),
                    api_key=api_key.strip(),
                    model=model_name.strip(),
                )
                # Refresh the dropdown
                configs = controller.list_llm_configs()
                return f"✅ Configuration '{config_name}' saved!", gr.update(
                    choices=[""] + configs
                )
            except Exception as exc:
                LOGGER.error("Failed to save configuration", exc_info=exc)
                return f"❌ Error: {exc}", gr.update()

        def handle_load_config(config_name: str):
            """Load a saved configuration."""
            if not config_name or config_name == "":
                return "", "", "", "lmstudio"

            try:
                config = controller.load_llm_config(config_name)
                if config:
                    return (
                        config.get("endpoint", ""),
                        config.get("api_key", ""),
                        config.get("model", ""),
                        config.get("provider", "lmstudio"),
                    )
                else:
                    return "", "", "", "lmstudio"
            except Exception as exc:
                LOGGER.error("Failed to load configuration", exc_info=exc)
                return "", "", "", "lmstudio"

        def handle_delete_config(config_name: str):
            """Delete a saved configuration."""
            if not config_name or config_name == "":
                return "⚠️ Please select a configuration to delete.", gr.update()

            try:
                controller.delete_llm_config(config_name)
                configs = controller.list_llm_configs()
                return f"✅ Configuration '{config_name}' deleted!", gr.update(
                    choices=[""] + configs, value=""
                )
            except Exception as exc:
                LOGGER.error("Failed to delete configuration", exc_info=exc)
                return f"❌ Error: {exc}", gr.update()

        # Build the UI
        with gr.Blocks() as module_ui:
            gr.Markdown("### Video LLM Interaction")
            gr.Markdown(
                "Upload a video clip and ask questions about its content. "
                "The LLM will analyze the video and provide answers in real-time."
            )

            # Configuration Section
            with gr.Accordion("LLM Configuration", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Saved configurations dropdown
                        saved_configs = controller.list_llm_configs()
                        config_selector = gr.Dropdown(
                            label="Load Saved Configuration",
                            choices=[""] + saved_configs,
                            value="",
                            info="Select a previously saved configuration",
                        )

                        # LLM provider selector
                        llm_provider_selector = gr.Radio(
                            label="LLM Provider",
                            choices=[
                                ("LM Studio (Local)", "lmstudio"),
                                ("Hugging Face (Cloud)", "huggingface"),
                            ],
                            value="lmstudio",
                            info="Choose the LLM inference provider",
                        )

                    with gr.Column(scale=3):
                        # Endpoint configuration
                        endpoint_url_input = gr.Textbox(
                            label="LLM Endpoint URL",
                            value="http://192.168.86.29:1234",
                            placeholder="http://192.168.86.29:1234 or HF endpoint",
                            info="Base URL for the LLM API",
                        )

                        api_key_input = gr.Textbox(
                            label="API Key",
                            value="lm-studio",
                            placeholder="Enter API key (if required)",
                            type="password",
                            info="API key for authentication",
                        )

                        model_name_input = gr.Textbox(
                            label="Model Name",
                            value="qwen/qwen2.5-vl-7b",
                            placeholder="qwen/qwen2.5-vl-7b or model path",
                            info="Name or path of the vision-language model",
                        )

                with gr.Row():
                    config_name_input = gr.Textbox(
                        label="Configuration Name",
                        placeholder="e.g., 'Local LM Studio', 'HF Production'",
                        scale=3,
                    )
                    save_config_btn = gr.Button("Save Config", variant="primary", scale=1)
                    delete_config_btn = gr.Button("Delete Config", variant="secondary", scale=1)

                config_feedback = gr.Markdown("")

            # Video Quality Controls
            with gr.Accordion("Video Quality & Processing Settings", open=False):
                with gr.Row():
                    num_frames_slider = gr.Slider(
                        label="Number of Frames to Sample",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        info="More frames = better context but slower processing",
                    )

                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        info="Randomness of LLM output (0=deterministic, higher=creative)",
                    )

                with gr.Row():
                    frame_width_input = gr.Number(
                        label="Frame Width (px)",
                        value=512,
                        minimum=128,
                        maximum=1920,
                        info="Width to resize frames before sending to LLM",
                    )

                    frame_height_input = gr.Number(
                        label="Frame Height (px)",
                        value=384,
                        minimum=96,
                        maximum=1080,
                        info="Height to resize frames before sending to LLM",
                    )

                    max_tokens_input = gr.Number(
                        label="Max Tokens",
                        value=1024,
                        minimum=128,
                        maximum=4096,
                        info="Maximum length of LLM response",
                    )

            # Main Video Analysis Section
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
                            ("MediaPipe - Pose Estimation", "pose"),
                            ("None - LLM Only", "none"),
                        ],
                        value="none",
                        info="Select the computer vision model for video preprocessing",
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
                    endpoint_url_input,
                    api_key_input,
                    model_name_input,
                    num_frames_slider,
                    frame_width_input,
                    frame_height_input,
                    temperature_slider,
                    max_tokens_input,
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

            # Configuration management handlers
            save_config_btn.click(
                handle_save_config,
                inputs=[
                    config_name_input,
                    endpoint_url_input,
                    api_key_input,
                    model_name_input,
                    llm_provider_selector,
                ],
                outputs=[config_feedback, config_selector],
            )

            config_selector.change(
                handle_load_config,
                inputs=[config_selector],
                outputs=[
                    endpoint_url_input,
                    api_key_input,
                    model_name_input,
                    llm_provider_selector,
                ],
            )

            delete_config_btn.click(
                handle_delete_config,
                inputs=[config_selector],
                outputs=[config_feedback, config_selector],
            )

        return module_ui


__all__ = ["VideoLLMInteractionModule"]
