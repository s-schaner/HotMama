from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List

import gradio as gr

from app.bus import EventBus
from app.registry import discover_plugins
from plugins.plugins_api import AppContext
from session.schemas import ClipCreate, EventIn, RosterEntry, SessionCreate
from session.service import SessionService
from viz.heatmap import available_renderers, render_with_strategy
from viz.trajectory import TrajectoryError, list_detection_methods, run_heatmap_pipeline


class GradioVolleyApp:
    def __init__(
        self,
        service: SessionService,
        sessions_dir: Path,
        plugins_dir: Path,
        bus: EventBus,
    ) -> None:
        self.service = service
        self.sessions_dir = sessions_dir
        self.bus = bus
        ctx = AppContext(
            db_url=str(service.engine.url), sessions_dir=sessions_dir, config={}
        )
        self.plugins = discover_plugins(plugins_dir, ctx)
        self.renderer_names = list(available_renderers().keys()) or ["basic"]
        if "basic" not in self.renderer_names:
            self.renderer_names.insert(0, "basic")
        methods = list_detection_methods()
        if "auto" in methods:
            methods = ["auto"] + [m for m in methods if m != "auto"]
        self.detection_methods = methods

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
                heatmap_session = gr.Textbox(
                    label="Session ID (optional)",
                    placeholder="Outputs saved under this session when provided",
                )

                gr.Markdown("### Video → Heatmap Pipeline")
                pipeline_video = gr.File(
                    label="Video", file_types=["video"], type="filepath"
                )
                pipeline_method = gr.Dropdown(
                    choices=self.detection_methods,
                    value=(
                        self.detection_methods[0] if self.detection_methods else "auto"
                    ),
                    label="Detection method",
                )
                with gr.Row():
                    pipeline_stride = gr.Slider(
                        1, 8, value=2, step=1, label="Frame stride"
                    )
                    pipeline_smoothing = gr.Slider(
                        1, 15, value=3, step=1, label="Smoothing window"
                    )
                with gr.Row():
                    pipeline_min_conf = gr.Slider(
                        0.0, 1.0, value=0.05, step=0.01, label="Min confidence"
                    )
                    pipeline_overlay = gr.Checkbox(
                        value=True, label="Generate overlays"
                    )
                pipeline_renderers = gr.CheckboxGroup(
                    choices=self.renderer_names,
                    value=self.renderer_names,
                    label="Heatmap renderers",
                )
                pipeline_run = gr.Button("Run Heatmap Pipeline")
                pipeline_status = gr.Markdown()
                pipeline_gallery = gr.Gallery(
                    label="Heatmap renders", show_label=True, columns=3
                )
                pipeline_csv = gr.File(label="Ball trajectory CSV")
                pipeline_overlay_video = gr.Video(label="Overlay video")
                pipeline_topdown = gr.Video(label="Court playback")
                pipeline_snapshot = gr.Image(label="Trajectory snapshot")
                pipeline_logs = gr.Markdown(label="Logs")

                gr.Markdown("### Manual CSV Renderer")
                csv_points = gr.Textbox(
                    label="CSV points x,y,h per line", value="1,1,0\n2,2,0.5"
                )
                manual_renderer = gr.Dropdown(
                    choices=self.renderer_names,
                    value=self.renderer_names[0],
                    label="Renderer",
                )
                render_btn = gr.Button("Render Heatmap")
                heatmap_image = gr.Image(label="Heatmap")

            for plugin in self.plugins.values():
                if hasattr(plugin, "get_ui_blocks"):
                    try:
                        plugin.get_ui_blocks()
                    except Exception:
                        gr.Markdown(f"Plugin {plugin.name} failed to create UI")

            create_btn.click(
                self._create_session,
                [session_title, session_venue, session_meta],
                sessions_out,
            )
            roster_btn.click(
                self._set_roster,
                [roster_session, roster_side, roster_json],
                roster_status,
            )
            clip_btn.click(
                self._add_clip,
                [clip_session, clip_path, clip_duration, clip_meta],
                clip_id_out,
            )
            events_btn.click(
                self._add_events, [analysis_session, analysis_clip, events_json], None
            )
            rollup_btn.click(self._get_rollup, [analysis_session], rollup_out)
            pipeline_run.click(
                self._run_heatmap_pipeline,
                [
                    heatmap_session,
                    pipeline_video,
                    pipeline_method,
                    pipeline_stride,
                    pipeline_smoothing,
                    pipeline_min_conf,
                    pipeline_overlay,
                    pipeline_renderers,
                ],
                [
                    pipeline_status,
                    pipeline_gallery,
                    pipeline_csv,
                    pipeline_overlay_video,
                    pipeline_topdown,
                    pipeline_snapshot,
                    pipeline_logs,
                ],
            )
            render_btn.click(
                self._render_heatmap,
                [heatmap_session, csv_points, manual_renderer],
                heatmap_image,
            )

        return demo

    def _create_session(self, title: str, venue: str, meta_json: str) -> List[dict]:
        data = SessionCreate(
            title=title, venue=venue or None, meta=json.loads(meta_json or "{}")
        )
        session_id = self.service.create_session(data)
        self.bus.emit("session_open", session_id=session_id)
        return [s.dict() for s in self.service.list_sessions()]

    def _set_roster(self, session_id: str, side: str, roster_json: str) -> str:
        entries = [RosterEntry(**item) for item in json.loads(roster_json or "[]")]
        self.service.set_roster(session_id, side, entries)
        return "Roster updated"

    def _add_clip(
        self, session_id: str, path: str, duration: float, meta_json: str
    ) -> str:
        clip_id = self.service.add_clip(
            session_id,
            ClipCreate(
                path=path,
                duration_sec=float(duration),
                meta=json.loads(meta_json or "{}"),
            ),
        )
        self.bus.emit(
            "clip_ingested", session_id=session_id, clip_id=clip_id, clip_path=path
        )
        return clip_id

    def _add_events(self, session_id: str, clip_id: str, events_json: str) -> None:
        events = [EventIn(**item) for item in json.loads(events_json or "[]")]
        self.service.add_events(session_id, clip_id, events)
        self.bus.emit(
            "events_parsed",
            session_id=session_id,
            clip_id=clip_id,
            events=[e.dict() for e in events],
        )

    def _get_rollup(self, session_id: str) -> List[dict]:
        return [r.dict() for r in self.service.get_rollup(session_id)]

    def _render_heatmap(self, session_id: str, csv_points: str, renderer_name: str):
        lines = [line.strip() for line in csv_points.splitlines() if line.strip()]
        points = []
        for line in lines:
            x_str, y_str, h_str = line.split(",")
            points.append((float(x_str), float(y_str), float(h_str)))
        session_key = session_id.strip() or "adhoc"
        out_dir = self.sessions_dir / session_key / "heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        renderer = renderer_name or "basic"
        out_path = out_dir / f"manual_{renderer}.png"
        render_with_strategy(renderer, points, out_path)
        return str(out_path)

    def _run_heatmap_pipeline(
        self,
        session_id: str,
        video_path: str | None,
        method: str,
        stride: float,
        smoothing: float,
        min_confidence: float,
        overlay: bool,
        renderers: List[str] | None,
    ):
        if not video_path:
            return (
                "⚠️ Provide a video file to run the pipeline.",
                [],
                None,
                None,
                None,
                None,
                "No video supplied.",
            )

        session_key = session_id.strip() or "adhoc"
        run_dir = (
            self.sessions_dir / session_key / "heatmaps" / f"run_{uuid.uuid4().hex}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            artifacts = run_heatmap_pipeline(
                video_path,
                run_dir,
                method=method or "auto",
                stride=int(max(1, stride)),
                smoothing_window=int(max(1, smoothing)),
                min_confidence=float(min_confidence),
                overlay=bool(overlay),
                heatmap_strategies=renderers or self.renderer_names,
            )
        except TrajectoryError as exc:
            return (f"❌ {exc}", [], None, None, None, None, str(exc))
        except Exception as exc:
            return (
                f"❌ Pipeline failed: {exc}",
                [],
                None,
                None,
                None,
                None,
                str(exc),
            )

        status = f"✅ Generated {len(artifacts.points)} points via {artifacts.method}"
        gallery = [
            (str(path), name) for name, path in sorted(artifacts.heatmaps.items())
        ]
        logs = (
            "\n".join(artifacts.logs)
            if artifacts.logs
            else "Pipeline completed without warnings."
        )
        return (
            status,
            gallery,
            str(artifacts.csv_path) if artifacts.csv_path else None,
            str(artifacts.overlay_video) if artifacts.overlay_video else None,
            str(artifacts.court_track_video) if artifacts.court_track_video else None,
            str(artifacts.trail_image) if artifacts.trail_image else None,
            logs,
        )
