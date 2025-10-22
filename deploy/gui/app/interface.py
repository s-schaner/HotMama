"""Gradio interface composition."""

from __future__ import annotations

import json
from html import escape

from deploy.api.app.schemas import JobSpec

from .controller import GuiController

CUSTOM_CSS = """
body {
    background-color: #e5e7eb;
    color: #111827;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.6;
}

.gradio-container {
    max-width: 1140px !important;
    margin: 0 auto;
    padding: 2.25rem 1.75rem 3rem;
    background-color: #f8fafc;
    color: inherit;
}

.hero {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 1rem;
    padding: 2rem;
    color: #111827;
}

.hero__eyebrow {
    display: inline-block;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    color: #1d4ed8;
    margin-bottom: 0.85rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.25rem;
    font-weight: 700;
    line-height: 1.2;
    color: #0f172a;
}

.hero p {
    margin-top: 0.75rem;
    max-width: 640px;
    color: #374151;
    font-size: 1.05rem;
}

.feature-grid {
    margin-top: 1.75rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
}

.feature {
    padding: 1.25rem 1.4rem;
    border-radius: 0.9rem;
    border: 1px solid #d1d5db;
    background-color: #ffffff;
    color: #1f2937;
}

.feature__eyebrow {
    display: inline-block;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    color: #1d4ed8;
    margin-bottom: 0.5rem;
}

.feature h3 {
    margin: 0 0 0.35rem 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: #0f172a;
}

.feature__body {
    margin: 0;
    color: #374151;
    font-size: 0.95rem;
}

.main-grid {
    margin-top: 2.25rem;
    gap: 1.5rem;
}

.card {
    border-radius: 1rem;
    padding: 1.75rem;
    border: 1px solid #d1d5db;
    background-color: #ffffff;
    color: #1f2937;
}

.card h3 {
    margin-top: 0;
    font-size: 1.5rem;
    font-weight: 600;
    color: #0f172a;
}

.card__subtitle {
    margin-bottom: 1.25rem;
    color: #4b5563;
}

.card--status {
    background-color: #0f172a;
    border-color: #1f2937;
    color: #f8fafc;
}

.card--status h3 {
    color: #f8fafc;
}

.card--status .card__subtitle {
    color: #cbd5f5;
}

.alert {
    margin-top: 1rem;
    border-radius: 0.75rem;
    padding: 0.9rem 1rem;
    font-weight: 600;
    border: 1px solid transparent;
}

.alert--success {
    background-color: #ecfdf3;
    border-color: #0f5132;
    color: #0f5132;
}

.alert--warning {
    background-color: #fef3c7;
    border-color: #92400e;
    color: #92400e;
}

.alert--danger {
    background-color: #fee2e2;
    border-color: #7f1d1d;
    color: #7f1d1d;
}

.alert--info {
    background-color: #dbeafe;
    border-color: #1d4ed8;
    color: #1d4ed8;
}

.status-json {
    max-height: 320px;
    overflow-y: auto;
    border-radius: 0.75rem;
    border: 1px solid #1f2937;
}

.status-json > div {
    background-color: #111827;
}

.status-json pre {
    color: #f8fafc !important;
}

.video-preview {
    margin-top: 1.25rem;
}

.video-preview video {
    width: 100%;
    border-radius: 0.75rem;
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
}

.gradio-button.primary {
    font-weight: 600;
    background-color: #1f2937;
    border: 1px solid #0f172a;
    color: #f8fafc;
}

.gradio-button.secondary {
    font-weight: 600;
    border: 1px solid #1f2937;
    color: #1f2937;
    background-color: transparent;
}

.hero code,
.feature__body code,
.form-footer code {
    background-color: #e5e7eb;
    color: #111827;
    padding: 0.1rem 0.3rem;
    border-radius: 0.25rem;
    font-size: 0.85em;
}

.form-footer {
    margin-top: 1.25rem;
    font-size: 0.9rem;
    color: #4b5563;
}

.card--status .form-footer {
    color: #e2e8f0;
}
"""

HERO_HTML = """
<div class="hero">
  <span class="hero__eyebrow">Operations Console</span>
  <h1>HotMama Vision Console</h1>
  <p>Queue new computer-vision runs, watch the refreshed worker pipeline in motion, and retrieve artifacts without leaving the browser.</p>
</div>
"""

FEATURE_GRID_HTML = """
<div class="feature-grid">
  <div class="feature">
    <span class="feature__eyebrow">Gateway</span>
    <h3>FastAPI v1</h3>
    <p class="feature__body">Submissions flow straight into the Redis queue module we just refactored for resiliency.</p>
  </div>
  <div class="feature">
    <span class="feature__eyebrow">Workers</span>
    <h3>Vision Runtime</h3>
    <p class="feature__body">The torch + OpenCV stack ingests uploads and produces session artifacts with deterministic seeds.</p>
  </div>
  <div class="feature">
    <span class="feature__eyebrow">Storage</span>
    <h3>Shared Sessions</h3>
    <p class="feature__body">Artifacts land in <code>/sessions</code> so the GUI can serve them back instantly after completion.</p>
  </div>
</div>
"""

JOB_SCHEMA_JSON = json.dumps(JobSpec.model_json_schema(), indent=2, sort_keys=True)
DEFAULT_INPUT_MODE = "json"

HELP_GUIDE_MARKDOWN = """
**Job manifest guidance**

- Use **Structured JSON** mode to paste an entire job manifest or simple option overrides.
- Use **Natural language** mode to describe the run. The assistant converts the request to the manifest shown below.
- The overlay selections above are merged into the manifest automatically before queueing.
""".strip()


def create_interface(controller: GuiController):
    """Construct and return the Gradio Blocks layout with modular tabs."""

    import gradio as gr

    from .modules import VideoLLMInteractionModule

    def _render_alert(message: str) -> str:
        if not message:
            return ""
        tone = "info"
        if message.startswith("✅"):
            tone = "success"
        elif message.startswith("⚠️"):
            tone = "warning"
        elif message.startswith("❌"):
            tone = "danger"
        return f"<div class='alert alert--{tone}'>{escape(message)}</div>"

    def _video_value(path: str | None):
        if not path:
            return gr.update(value=None)
        return str(path)

    def _manifest_update(manifest: dict | None):
        if not manifest:
            return gr.update(value={}, visible=False)
        return gr.update(value=manifest, visible=True)

    def _handle_submit(
        upload_path: str | None,
        params_text: str,
        profile_value: str,
        overlay_values: list[str] | None,
        current_job_id: str,
        current_monitor: str,
        input_mode_value: str,
        prompt_text: str,
        enrich_value: bool,
    ):
        job_id, message, manifest = controller.submit_job(
            upload_path,
            params_text,
            job_type=profile_value,
            overlay_modes=overlay_values or [],
            input_mode=input_mode_value,
            prompt_text=prompt_text,
            enrich_manifest=enrich_value,
        )
        banner = _render_alert(message)
        source_preview = _video_value(upload_path)
        manifest_preview = _manifest_update(manifest)
        if not job_id:
            return (
                gr.update(value=current_job_id),
                gr.update(value=current_monitor),
                banner,
                "",
                gr.update(),
                gr.update(),
                gr.update(value=None),
                source_preview,
                current_job_id,
                manifest_preview,
            )
        return (
            job_id,
            job_id,
            banner,
            "",
            {},
            gr.update(value=None),
            gr.update(value=None),
            source_preview,
            job_id,
            manifest_preview,
        )

    def _handle_refresh(job_id_text: str, current_active: str):
        status, message, artifact = controller.refresh_status(job_id_text)
        banner = _render_alert(message)
        artifact_value = artifact or None
        artifact_preview = _video_value(artifact_value)
        if status.get("status"):
            active = job_id_text
            job_display = job_id_text
        elif current_active:
            active = current_active
            job_display = current_active
        else:
            active = ""
            job_display = ""
        return (
            job_display,
            status,
            banner,
            artifact_value,
            artifact_preview,
            active,
        )

    def _handle_source_preview(upload_path: str | None):
        return _video_value(upload_path)

    def _handle_mode_change(mode: str):
        is_nl = mode == "nl"
        return (
            gr.update(visible=not is_nl),
            gr.update(visible=is_nl),
            gr.update(visible=is_nl),
        )

    def _toggle_help(is_open: bool):
        if is_open:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                False,
                gr.update(value="Show schema"),
            )
        return (
            gr.update(value=HELP_GUIDE_MARKDOWN, visible=True),
            gr.update(value=JOB_SCHEMA_JSON, visible=True),
            True,
            gr.update(value="Hide schema"),
        )

    def _handle_clear():
        return (
            gr.update(value=None),
            gr.update(value="{}", visible=True),
            "",
            gr.update(value=None),
            "pipeline.analysis",
            ["overlay.activity_heatmap"],
            gr.update(value={}, visible=False),
            DEFAULT_INPUT_MODE,
            gr.update(value="", visible=False),
            gr.update(value=False, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            False,
            gr.update(value="Show schema"),
        )

    with gr.Blocks(title="HotMama Vision Console", css=CUSTOM_CSS) as demo:
        gr.HTML(HERO_HTML)
        gr.HTML(FEATURE_GRID_HTML)

        # Tab-based interface for different modules
        with gr.Tabs():
            with gr.Tab("Video Analysis"):
                active_job = gr.State("")

                with gr.Row(elem_classes=["main-grid"], equal_height=True):
                    with gr.Column(elem_classes=["card", "card--form"], scale=6):
                        gr.Markdown("### Queue a new analysis run")
                        gr.Markdown(
                            "Submit a short clip and optional JSON parameters. We'll hand it off to the refreshed FastAPI + Redis queue stack.",
                            elem_classes=["card__subtitle"],
                        )
                        upload = gr.File(
                            label="Vision clip",
                            file_types=[".mp4", ".mov", ".mkv", ".avi"],
                            type="filepath",
                        )
                        source_preview = gr.Video(
                            label="Source preview",
                            interactive=False,
                            height=360,
                            elem_classes=["video-preview"],
                            format=None,
                        )
                        profile = gr.Radio(
                            label="Analysis profile",
                            choices=[
                                ("Full analysis run", "pipeline.analysis"),
                                ("Segmentation sweep", "pipeline.segment"),
                                ("Event timeline triage", "pipeline.events"),
                            ],
                            value="pipeline.analysis",
                        )
                        overlays = gr.CheckboxGroup(
                            label="Video overlays",
                            choices=[
                                (
                                    "Activity heatmap (intensity map)",
                                    "overlay.activity_heatmap",
                                ),
                                (
                                    "Object tracking trails",
                                    "overlay.object_tracking",
                                ),
                                ("Pose skeletons", "overlay.pose_skeleton"),
                            ],
                            value=["overlay.activity_heatmap"],
                            info="Combine overlays to tailor the rendered diagnostic layers.",
                        )
                        mode_choices: list[tuple[str, str]] = [("Structured JSON", "json")]
                        if controller.supports_natural_language:
                            mode_choices.append(("Natural language", "nl"))
                        input_mode = gr.Radio(
                            label="Submission mode",
                            choices=mode_choices,
                            value=DEFAULT_INPUT_MODE,
                        )
                        params = gr.Code(
                            label="Parameters (JSON)",
                            value="{}",
                            language="json",
                            visible=True,
                        )
                        nl_prompt = gr.TextArea(
                            label="Natural language request",
                            placeholder="Describe the analysis you need, relevant moments, and desired overlays.",
                            visible=False,
                            lines=6,
                        )
                        enrich_checkbox = gr.Checkbox(
                            label="Enrich manifest with qwen2.5-vl-7b",
                            value=False,
                            visible=False,
                        )
                        help_state = gr.State(False)
                        with gr.Row():
                            help_btn = gr.Button(
                                "Show schema",
                                variant="secondary",
                                icon="❓",
                            )
                        help_text = gr.Markdown(visible=False)
                        help_schema = gr.Code(
                            label="Job manifest schema",
                            value="",
                            language="json",
                            interactive=False,
                            visible=False,
                        )
                        with gr.Row():
                            submit_btn = gr.Button(
                                "Queue analysis",
                                variant="primary",
                                icon="🚀",
                            )
                            clear_btn = gr.Button(
                                "Reset form",
                                variant="secondary",
                                icon="🧹",
                            )
                        job_id_display = gr.Textbox(
                            label="Active job",
                            interactive=False,
                            placeholder="Submit a job to populate",
                        )
                        submission_feedback = gr.HTML()
                        manifest_preview = gr.JSON(
                            label="Job manifest (preview)",
                            value={},
                            visible=False,
                        )
                        gr.Markdown(
                            "The upload is stored under <code>sessions/gui/uploads</code> and shared with the worker container.",
                            elem_classes=["form-footer"],
                        )

                    with gr.Column(elem_classes=["card", "card--status"], scale=6):
                        gr.Markdown("### Monitor progress")
                        gr.Markdown(
                            "Paste any job ID to follow along. Status updates stream from the Redis status hashes maintained by the worker module.",
                            elem_classes=["card__subtitle"],
                        )
                        job_id_input = gr.Textbox(
                            label="Job to monitor",
                            placeholder="00000000-0000-0000-0000-000000000000",
                        )
                        refresh_btn = gr.Button(
                            "Refresh status",
                            variant="primary",
                            icon="🔄",
                        )
                        status_message = gr.HTML()
                        status_json = gr.JSON(
                            label="Status detail",
                            value={},
                            elem_classes=["status-json"],
                        )
                        artifact_preview = gr.Video(
                            label="Processed preview",
                            interactive=False,
                            height=360,
                            elem_classes=["video-preview"],
                            format=None,
                        )
                        artifact_file = gr.File(
                            label="Artifact download",
                            interactive=False,
                        )
                        gr.Markdown(
                            "Artifacts are mirrored into <code>sessions/gui/downloads</code> for instant download once the worker reports completion.",
                            elem_classes=["form-footer"],
                        )

                # Event handlers for Video Analysis tab
                input_mode.change(
                    _handle_mode_change,
                    inputs=input_mode,
                    outputs=[params, nl_prompt, enrich_checkbox],
                )

                help_btn.click(
                    _toggle_help,
                    inputs=help_state,
                    outputs=[help_text, help_schema, help_state, help_btn],
                )

                upload.upload(
                    _handle_source_preview,
                    inputs=upload,
                    outputs=source_preview,
                )

                submit_btn.click(
                    _handle_submit,
                    inputs=[
                        upload,
                        params,
                        profile,
                        overlays,
                        active_job,
                        job_id_input,
                        input_mode,
                        nl_prompt,
                        enrich_checkbox,
                    ],
                    outputs=[
                        job_id_display,
                        job_id_input,
                        submission_feedback,
                        status_message,
                        status_json,
                        artifact_file,
                        artifact_preview,
                        source_preview,
                        active_job,
                        manifest_preview,
                    ],
                )

                refresh_btn.click(
                    _handle_refresh,
                    inputs=[job_id_input, active_job],
                    outputs=[
                        job_id_display,
                        status_json,
                        status_message,
                        artifact_file,
                        artifact_preview,
                        active_job,
                    ],
                )

                clear_btn.click(
                    _handle_clear,
                    inputs=None,
                    outputs=[
                        upload,
                        params,
                        submission_feedback,
                        source_preview,
                        profile,
                        overlays,
                        manifest_preview,
                        input_mode,
                        nl_prompt,
                        enrich_checkbox,
                        help_text,
                        help_schema,
                        help_state,
                        help_btn,
                    ],
                )

            # Video LLM Interaction Tab
            with gr.Tab("Video LLM Interaction"):
                video_llm_module = VideoLLMInteractionModule()
                video_llm_ui = video_llm_module.build_ui(controller)

    return demo


__all__ = ["create_interface"]
