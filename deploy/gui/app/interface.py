"""Gradio interface composition."""

from __future__ import annotations

from .controller import GuiController


def create_interface(controller: GuiController):
    """Construct and return the Gradio Blocks layout."""

    import gradio as gr

    with gr.Blocks(title="HotMama Control Center") as demo:
        gr.Markdown("""
        # HotMama Vision Control Center
        Submit processing jobs, monitor progress, and retrieve artifacts.
        """)

        with gr.Tab("Submit Job"):
            upload = gr.File(label="Video clip", file_types=[".mp4", ".mov", ".mkv", ".avi"], type="filepath")
            params = gr.Textbox(label="Parameters (JSON)", placeholder='{"stride": 2}', value="{}")
            submit_btn = gr.Button("Submit to queue")
            job_id_out = gr.Textbox(label="Job ID", interactive=False)
            submit_status = gr.Markdown()
            submit_btn.click(
                controller.submit_job,
                inputs=[upload, params],
                outputs=[job_id_out, submit_status],
            )

        with gr.Tab("Job Monitor"):
            job_id_input = gr.Textbox(label="Job ID")
            refresh_btn = gr.Button("Refresh status")
            status_json = gr.JSON(label="Status details")
            status_message = gr.Markdown()
            artifact_file = gr.File(label="Artifact", interactive=False)
            refresh_btn.click(
                controller.refresh_status,
                inputs=[job_id_input],
                outputs=[status_json, status_message, artifact_file],
            )

    return demo


__all__ = ["create_interface"]
