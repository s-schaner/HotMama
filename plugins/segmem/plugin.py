from __future__ import annotations

import gradio as gr

from plugins.plugins_api import HookResult


class DummyPlugin:
    name = "segmem"
    version = "0.1.0"
    provides_ui = True

    def on_register(self, ctx):
        self.ctx = ctx

    def on_session_open(self, session_id: str) -> None:
        pass

    def on_session_close(self, session_id: str) -> None:
        pass

    def on_clip_ingested(self, session_id: str, clip_id: str, clip_path: str) -> HookResult:
        return HookResult()

    def on_events_parsed(self, session_id: str, clip_id: str, events: list[dict]) -> HookResult:
        return HookResult()

    def get_ui_blocks(self):
        with gr.Tab("segmem"):
            gr.Markdown("Plugin segmem ready.")


PLUGIN = DummyPlugin()
