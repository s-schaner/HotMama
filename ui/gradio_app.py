from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import gradio as gr

from app.bus import EventBus
from app.registry import discover_plugins
from plugins.plugins_api import AppContext
from session.schemas import ClipCreate, EventIn, RosterEntry, SessionCreate
from session.service import SessionService
from viz.heatmap import render_heatmap


class GradioVolleyApp:
    def __init__(self, service: SessionService, sessions_dir: Path, plugins_dir: Path, bus: EventBus) -> None:
        self.service = service
        self.sessions_dir = sessions_dir
        self.bus = bus
        ctx = AppContext(db_url=str(service.engine.url), sessions_dir=sessions_dir, config={})
        self.plugins = discover_plugins(plugins_dir, ctx)

    def launch(self, share: bool = False, auth: tuple[str, str] | None = None):
        demo = self._build_interface()
        demo.launch(share=share, auth=auth)

    def _build_interface(self) -> gr.Blocks:
        with gr.Blocks(title="VolleySense") as demo:
            with gr.Tab("Sessions"):
                session_title = gr.Textbox(label="Title")
                session_venue = gr.Textbox(label="Venue")
                session_meta = gr.Textbox(label="Meta JSON", value="{}")
                create_btn = gr.Button("Create Session")
                sessions_out = gr.JSON(label="Sessions")

            with gr.Tab("Roster"):
                roster_session = gr.Textbox(label="Session ID")
                roster_side = gr.Radio(["A", "B"], label="Side", value="A")
                roster_json = gr.Textbox(label="Entries JSON", value="[]")
                roster_btn = gr.Button("Save Roster")
                roster_status = gr.Markdown()

            with gr.Tab("Clips"):
                clip_session = gr.Textbox(label="Session ID")
                clip_path = gr.Textbox(label="Clip Path")
                clip_duration = gr.Number(label="Duration", value=10.0)
                clip_meta = gr.Textbox(label="Meta JSON", value="{}")
                clip_btn = gr.Button("Add Clip")
                clip_id_out = gr.Textbox(label="Clip ID")

            with gr.Tab("Analysis"):
                analysis_session = gr.Textbox(label="Session ID")
                analysis_clip = gr.Textbox(label="Clip ID")
                events_json = gr.Textbox(label="Events JSON", value="[]")
                events_btn = gr.Button("Add Events")
                rollup_btn = gr.Button("Refresh Rollup")
                rollup_out = gr.JSON(label="Rollup")

            with gr.Tab("Heatmap"):
                heatmap_session = gr.Textbox(label="Session ID")
                csv_points = gr.Textbox(label="CSV points x,y,h per line", value="1,1,0\n2,2,0.5")
                render_btn = gr.Button("Render Heatmap")
                heatmap_image = gr.Image(label="Heatmap")

            for plugin in self.plugins.values():
                if hasattr(plugin, "get_ui_blocks"):
                    try:
                        plugin.get_ui_blocks()
                    except Exception:
                        gr.Markdown(f"Plugin {plugin.name} failed to create UI")

            create_btn.click(self._create_session, [session_title, session_venue, session_meta], sessions_out)
            roster_btn.click(self._set_roster, [roster_session, roster_side, roster_json], roster_status)
            clip_btn.click(self._add_clip, [clip_session, clip_path, clip_duration, clip_meta], clip_id_out)
            events_btn.click(self._add_events, [analysis_session, analysis_clip, events_json], None)
            rollup_btn.click(self._get_rollup, [analysis_session], rollup_out)
            render_btn.click(self._render_heatmap, [heatmap_session, csv_points], heatmap_image)

        return demo

    def _create_session(self, title: str, venue: str, meta_json: str) -> List[dict]:
        data = SessionCreate(title=title, venue=venue or None, meta=json.loads(meta_json or "{}"))
        session_id = self.service.create_session(data)
        self.bus.emit("session_open", session_id=session_id)
        return [s.dict() for s in self.service.list_sessions()]

    def _set_roster(self, session_id: str, side: str, roster_json: str) -> str:
        entries = [RosterEntry(**item) for item in json.loads(roster_json or "[]")]
        self.service.set_roster(session_id, side, entries)
        return "Roster updated"

    def _add_clip(self, session_id: str, path: str, duration: float, meta_json: str) -> str:
        clip_id = self.service.add_clip(session_id, ClipCreate(path=path, duration_sec=float(duration), meta=json.loads(meta_json or "{}")))
        self.bus.emit("clip_ingested", session_id=session_id, clip_id=clip_id, clip_path=path)
        return clip_id

    def _add_events(self, session_id: str, clip_id: str, events_json: str) -> None:
        events = [EventIn(**item) for item in json.loads(events_json or "[]")]
        self.service.add_events(session_id, clip_id, events)
        self.bus.emit("events_parsed", session_id=session_id, clip_id=clip_id, events=[e.dict() for e in events])

    def _get_rollup(self, session_id: str) -> List[dict]:
        return [r.dict() for r in self.service.get_rollup(session_id)]

    def _render_heatmap(self, session_id: str, csv_points: str):
        lines = [line.strip() for line in csv_points.splitlines() if line.strip()]
        points = []
        for line in lines:
            x_str, y_str, h_str = line.split(",")
            points.append((float(x_str), float(y_str), float(h_str)))
        out_dir = self.sessions_dir / session_id / "heatmaps"
        out_path = out_dir / "manual_heatmap.png"
        render_heatmap(points, out_path)
        return out_path
