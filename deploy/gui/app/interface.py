"""Gradio interface composition."""

from __future__ import annotations

from html import escape

from .controller import GuiController

CUSTOM_CSS = """
.gradio-container {
    max-width: 1140px !important;
    margin: 0 auto;
    padding: 2.5rem 1.75rem 3rem;
    background: radial-gradient(circle at 10% 20%, rgba(244,114,182,0.25), transparent 55%),
                radial-gradient(circle at 90% 10%, rgba(129,140,248,0.3), transparent 45%),
                linear-gradient(135deg, #020617 0%, #0f172a 45%, #1e293b 100%);
    color: #0f172a;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body {
    background-color: #020617;
}

.hero {
    background: linear-gradient(135deg, rgba(244,114,182,0.15), rgba(14,165,233,0.15));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 1.25rem;
    padding: 1.75rem 2rem;
    color: #f8fafc;
    box-shadow: 0 30px 80px rgba(15,23,42,0.35);
    backdrop-filter: blur(14px);
}

.hero h1 {
    margin: 0;
    font-size: 2.25rem;
    font-weight: 700;
}

.hero p {
    margin-top: 0.75rem;
    max-width: 640px;
    color: rgba(248,250,252,0.82);
    font-size: 1.05rem;
}

.feature-grid {
    margin-top: 1.75rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
}

.feature {
    padding: 1.25rem 1.4rem;
    border-radius: 1rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(15,23,42,0.6);
    color: #e2e8f0;
    backdrop-filter: blur(12px);
    box-shadow: 0 12px 35px rgba(15,23,42,0.35);
}

.feature__eyebrow {
    display: inline-block;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(244,114,182,0.7);
    margin-bottom: 0.5rem;
}

.feature h3 {
    margin: 0 0 0.35rem 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
}

.main-grid {
    margin-top: 2.25rem;
    gap: 1.5rem;
}

.card {
    border-radius: 1.3rem;
    padding: 1.75rem;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(248,250,252,0.88);
    box-shadow: 0 18px 45px rgba(15,23,42,0.35);
}

.card--status {
    background: rgba(12,18,32,0.82);
    color: #e2e8f0;
}

.card--status h3 {
    color: #f8fafc;
}

.card h3 {
    margin-top: 0;
    font-size: 1.5rem;
    font-weight: 600;
    color: #0f172a;
}

.card__subtitle {
    margin-bottom: 1.25rem;
    color: rgba(15,23,42,0.75);
}

.card--status .card__subtitle {
    color: rgba(226,232,240,0.75);
}

.alert {
    margin-top: 1rem;
    border-radius: 0.9rem;
    padding: 0.95rem 1.15rem;
    font-weight: 600;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
}

.alert--success {
    background: rgba(34,197,94,0.16);
    color: #14532d;
    border-color: rgba(74,222,128,0.4);
}

.alert--warning {
    background: rgba(250,204,21,0.18);
    color: #854d0e;
    border-color: rgba(250,204,21,0.45);
}

.alert--danger {
    background: rgba(248,113,113,0.2);
    color: #7f1d1d;
    border-color: rgba(248,113,113,0.45);
}

.alert--info {
    background: rgba(191,219,254,0.25);
    color: #1d4ed8;
    border-color: rgba(147,197,253,0.45);
}

.status-json {
    max-height: 320px;
    overflow-y: auto;
    border-radius: 0.9rem;
}

.status-json > div {
    background: rgba(15,23,42,0.92);
}

.status-json pre {
    color: #e2e8f0 !important;
}

.gradio-button.primary {
    font-weight: 600;
}

.form-footer {
    margin-top: 1.25rem;
    font-size: 0.9rem;
    color: rgba(15,23,42,0.6);
}

.card--status .form-footer {
    color: rgba(226,232,240,0.65);
}
"""

HERO_HTML = """
<div class="hero">
  <h1>HotMama Vision Console</h1>
  <p>Queue new computer-vision runs, watch the refreshed worker pipeline in motion, and retrieve artifacts without leaving the browser.</p>
</div>
"""

FEATURE_GRID_HTML = """
<div class="feature-grid">
  <div class="feature">
    <span class="feature__eyebrow">Gateway</span>
    <h3>FastAPI v1</h3>
    <p>Submissions flow straight into the Redis queue module we just refactored for resiliency.</p>
  </div>
  <div class="feature">
    <span class="feature__eyebrow">Workers</span>
    <h3>Vision Runtime</h3>
    <p>The torch + OpenCV stack ingests uploads and produces session artifacts with deterministic seeds.</p>
  </div>
  <div class="feature">
    <span class="feature__eyebrow">Storage</span>
    <h3>Shared Sessions</h3>
    <p>Artifacts land in <code>/sessions</code> so the GUI can serve them back instantly after completion.</p>
  </div>
</div>
"""


def create_interface(controller: GuiController):
    """Construct and return the Gradio Blocks layout."""

    import gradio as gr

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

    def _handle_submit(
        upload_path: str | None,
        params_text: str,
        job_type_value: str,
        current_job_id: str,
        current_monitor: str,
    ):
        job_id, message = controller.submit_job(
            upload_path, params_text, job_type=job_type_value
        )
        banner = _render_alert(message)
        if not job_id:
            return (
                gr.update(value=current_job_id),
                gr.update(value=current_monitor),
                banner,
                "",
                gr.update(),
                gr.update(),
                current_job_id,
            )
        return (
            job_id,
            job_id,
            banner,
            "",
            {},
            gr.update(value=None),
            job_id,
        )

    def _handle_refresh(job_id_text: str, current_active: str):
        status, message, artifact = controller.refresh_status(job_id_text)
        banner = _render_alert(message)
        artifact_value = artifact or None
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
            active,
        )

    with gr.Blocks(title="HotMama Vision Console", css=CUSTOM_CSS) as demo:
        gr.HTML(HERO_HTML)
        gr.HTML(FEATURE_GRID_HTML)

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
                job_type = gr.Dropdown(
                    label="Job type",
                    choices=[
                        ("Vision pipeline", "vision.process"),
                        ("Segmentation sweep", "vision.segment"),
                        ("Tracking diagnostics", "vision.track"),
                    ],
                    value="vision.process",
                )
                params = gr.Code(
                    label="Parameters (JSON)",
                    value="{}",
                    language="json",
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
                artifact_file = gr.File(
                    label="Artifact download",
                    interactive=False,
                )
                gr.Markdown(
                    "Artifacts are mirrored into <code>sessions/gui/downloads</code> for instant download once the worker reports completion.",
                    elem_classes=["form-footer"],
                )

        submit_btn.click(
            _handle_submit,
            inputs=[upload, params, job_type, active_job, job_id_input],
            outputs=[
                job_id_display,
                job_id_input,
                submission_feedback,
                status_message,
                status_json,
                artifact_file,
                active_job,
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
                active_job,
            ],
        )

        clear_btn.click(
            lambda: (gr.update(value=None), "{}", ""),
            inputs=None,
            outputs=[upload, params, submission_feedback],
        )

    return demo


__all__ = ["create_interface"]
