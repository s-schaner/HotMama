from __future__ import annotations

import contextlib
import csv
import json
import os
import uuid
from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from core.analysis import (
    b64_image_data_uri,
    call_endpoint_openai_chat,
    extract_frames,
    render_heatmap,
)
from viz.heatmap import available_renderers, render_heatmap_sexy
from viz.trajectory import (
    TrajectoryError,
    list_detection_methods,
    run_heatmap_pipeline,
)

from webapp.presets import EndpointPreset, load_presets, upsert_preset


SESSION_DIR = os.environ.get(
    "VOLLEYSENSE_SESSIONS", os.path.join(os.getcwd(), "sessions")
)
os.makedirs(SESSION_DIR, exist_ok=True)
PRESETS_PATH = Path(SESSION_DIR) / "endpoint_presets.json"
load_presets(PRESETS_PATH)


TOOLKIT_PROMPTS: dict[str, str] = {
    "identify_players": (
        "Identify every player visible on the court, including team association, jersey color, jersey number, and likely role/position."
    ),
    "generate_stats": (
        "Generate rally statistics summarizing touches (serve, set, attack, dig, block, freeball, etc.) per player and team with counts."
    ),
    "build_heatmap": (
        "Ensure heatmap_points captures the ball trajectory for every touch using reliable detection or estimation if necessary."
    ),
    "pose_estimation": (
        "Describe pose/stance estimations for all players immediately before, during, and after each touch in the rally."
    ),
}


def _public_url(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        abs_root = Path(os.getcwd()).resolve()
        resolved = Path(path).resolve()
        rel = resolved.as_posix().replace(abs_root.as_posix(), "")
        return f"/static/..{rel}"
    except Exception:
        return None


class PresetPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    endpoint: str = Field(..., min_length=1)
    token: str = Field(default="")
    model: str = Field(..., min_length=1)

app = FastAPI(title="VolleySense")
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")


def _coerce_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "on", "yes"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    presets = [preset.to_dict() for preset in load_presets(PRESETS_PATH)]
    context = {
        "request": request,
        "presets": presets,
        "toolkit_prompts": TOOLKIT_PROMPTS,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/_partial/heatmap", response_class=HTMLResponse)
async def partial_heatmap(request: Request) -> HTMLResponse:
    renderers = list(available_renderers().keys()) or ["basic"]
    detections = list_detection_methods()
    context = {
        "request": request,
        "renderers": renderers,
        "detection_methods": detections,
    }
    return templates.TemplateResponse("_heatmap_panel.html", context)


@app.get("/_partial/sessions", response_class=HTMLResponse)
async def partial_sessions(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_sessions_panel.html", {"request": request})


@app.post("/_api/analyze", response_class=HTMLResponse)
async def analyze_htmx(
    request: Request,
    video: UploadFile = File(...),
    prompt: str = Form(
        "describe the visualized plays, describe each point, the sequence of events in the point, the touches in the point, broken down by who (color and roster number) on each team did what."
    ),
    endpoint: str = Form(...),
    token: str = Form(""),
    model: str = Form("Qwen/Qwen2.5-VL-32B-Instruct"),
    fps: int = Form(3),
    max_frames: int = Form(48),
    max_pixels: int = Form(1048576),
    max_tokens: int = Form(256),
    temperature: float = Form(0.2),
    jpeg_quality: int = Form(85),
    tools: List[str] = Form([]),
):
    data = await video.read()
    tmp_path = os.path.join(SESSION_DIR, f"upload_{video.filename}")
    with open(tmp_path, "wb") as fh:
        fh.write(data)

    try:
        frames, meta = extract_frames(tmp_path, float(fps), int(max_frames))
        system_prompt = open("prompts/qwen_volley_system.txt").read()
        directives = []
        for tool in tools or []:
            text = TOOLKIT_PROMPTS.get(tool)
            if text and text not in directives:
                directives.append(text)
        if directives:
            tool_text = "\n".join(f"- {line}" for line in directives)
            prompt = prompt.rstrip() + "\n\nAnalysis directives:\n" + tool_text
        images = [b64_image_data_uri(frame, q=int(jpeg_quality)) for frame in frames]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in images
                ],
            },
        ]

        response = call_endpoint_openai_chat(
            endpoint,
            token or None,
            model,
            messages,
            fps=fps,
            max_px=max_pixels,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = json.dumps(response, indent=2)
        pretty = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        choice = (
            (response.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        try:
            start = choice.find("{")
            end = choice.rfind("}")
            payload = choice if start == -1 else choice[start : end + 1]
            model_json = json.loads(payload)
        except Exception:
            model_json = None

        heatmap_img_path = None
        heatmap_csv_path = None
        if model_json and isinstance(model_json, dict):
            pts = model_json.get("heatmap_points") or []
            valid_pts = []
            for p in pts:
                try:
                    x, y, h = float(p["x"]), float(p["y"]), float(p.get("h", 0))
                    if 0 <= x <= 9 and 0 <= y <= 18:
                        valid_pts.append((x, y, h))
                except Exception:
                    continue
            if valid_pts:
                out_dir = os.path.join(SESSION_DIR, "heatmaps")
                os.makedirs(out_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(video.filename))[0]
                heatmap_img_path = render_heatmap(
                    valid_pts,
                    os.path.join(
                        out_dir, f"hm_{base_name}.png"
                    ),
                )
                csv_path = Path(out_dir) / f"hm_{base_name}.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["x", "y", "h"])
                    for x, y, h in valid_pts:
                        writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{h:.4f}"])
                heatmap_csv_path = csv_path

        context = {
            "request": request,
            "pretty": pretty or "(No content)",
            "raw": raw,
            "meta": meta,
            "tools": [TOOLKIT_PROMPTS.get(t, t) for t in tools or []],
        }
        html = templates.get_template("_result_panel.html").render(context)
        if heatmap_img_path:
            html += (
                "<div class=\"divider\">Heatmap</div>"
                f"<img src=\"/static/..{heatmap_img_path.replace(os.getcwd(), '')}\" "
                "class=\"rounded border border-base-300\">"
            )
            if heatmap_csv_path:
                csv_url = _public_url(heatmap_csv_path)
                if csv_url:
                    html += (
                        "<p class=\"mt-2 text-sm\">Ball trajectory CSV: "
                        f"<a class=\"link link-primary\" href=\"{csv_url}\" target=\"_blank\">download</a></p>"
                    )
        return HTMLResponse(html)
    except Exception as exc:
        return HTMLResponse(
            f"<div class='alert alert-error'><span>Error: {exc}</span></div>",
            status_code=500,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/_api/heatmap_pipeline", response_class=HTMLResponse)
async def heatmap_pipeline(
    request: Request,
    video: UploadFile = File(...),
    method: str = Form("auto"),
    stride: int = Form(2),
    smoothing: int = Form(3),
    min_confidence: float = Form(0.05),
    overlay: str = Form("true"),
    renderers: List[str] | None = Form(None),
) -> HTMLResponse:
    temp_path = Path(SESSION_DIR) / f"pipeline_{uuid.uuid4().hex}_{video.filename}"
    temp_path.write_bytes(await video.read())

    run_dir = Path(SESSION_DIR) / "heatmaps" / f"run_{uuid.uuid4().hex}"
    try:
        selected_renderers = [r for r in (renderers or []) if r]
        artifacts = run_heatmap_pipeline(
            temp_path,
            run_dir,
            method=method,
            stride=int(stride),
            smoothing_window=int(smoothing),
            min_confidence=float(min_confidence),
            overlay=_coerce_bool(overlay),
            heatmap_strategies=selected_renderers or ["basic", "opencv", "matplotlib"],
        )
    except TrajectoryError as exc:
        html = f"<div class='alert alert-error'><span>{exc}</span></div>"
        return HTMLResponse(html, status_code=400)
    except Exception as exc:
        html = f"<div class='alert alert-error'><span>Pipeline failed: {exc}</span></div>"
        return HTMLResponse(html, status_code=500)
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)

    heatmap_entries = []
    for name, path in sorted(artifacts.heatmaps.items()):
        url = _public_url(path)
        if url:
            heatmap_entries.append({"name": name, "url": url})

    context = {
        "request": request,
        "method": artifacts.method,
        "points": artifacts.points,
        "points_count": len(artifacts.points),
        "csv_url": _public_url(artifacts.csv_path),
        "heatmaps": heatmap_entries,
        "overlay_url": _public_url(artifacts.overlay_video),
        "topdown_url": _public_url(artifacts.court_track_video),
        "snapshot_url": _public_url(artifacts.trail_image),
        "logs": artifacts.logs,
        "preview": [
            {
                "t": f"{pt.t:.2f}",
                "x": f"{pt.x:.2f}",
                "y": f"{pt.y:.2f}",
                "h": f"{pt.h:.2f}",
                "confidence": f"{pt.confidence:.2f}",
                "frame": pt.frame_index,
            }
            for pt in artifacts.points[:10]
        ],
    }
    html = templates.get_template("_heatmap_pipeline_result.html").render(context)
    return HTMLResponse(html)


@app.post("/heatmap")
async def heatmap(
    request: Request,
    csvfile: UploadFile = File(...),
    theme: str = Form("dark"),
    colormap: str = Form("magma"),
    density_scale: float = Form(1.0),
    density_sigma: float = Form(1.0),
    show_contours: str = Form("True"),
    draw_trail: str = Form("True"),
    trail_width_px: float = Form(10.0),
) -> FileResponse:
    _ = request
    data = (await csvfile.read()).decode("utf-8").splitlines()
    reader = csv.reader(data)

    points: List[Tuple[float, float, float]] = []
    for i, row in enumerate(reader):
        if i == 0 and any(val.lower() in {"x", "y", "h"} for val in row):
            continue
        if len(row) < 3:
            continue
        points.append((float(row[0]), float(row[1]), float(row[2])))

    out_path = os.path.join(
        SESSION_DIR,
        "heatmaps",
        f"heatmap_{os.path.basename(csvfile.filename)}.png",
    )
    try:
        saved_path = render_heatmap_sexy(
            points,
            out_path,
            theme="dark" if theme not in {"dark", "light"} else theme,
            colormap=colormap or "magma",
            density_scale=float(density_scale),
            density_sigma_mul=float(density_sigma),
            show_contours=_coerce_bool(show_contours),
            draw_trail=_coerce_bool(draw_trail),
            trail_width_px=float(trail_width_px),
        )
    except Exception:
        saved_path = Path(render_heatmap(points, out_path))

    return FileResponse(saved_path, media_type="image/png")


def run(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run("webapp.server:app", host=host, port=port, reload=False)
@app.get("/_api/presets")
async def list_presets() -> dict:
    presets = [preset.to_dict() for preset in load_presets(PRESETS_PATH)]
    return {"presets": presets}


@app.post("/_api/presets")
async def create_preset(preset: PresetPayload) -> dict:
    name = preset.name.strip()
    endpoint = preset.endpoint.strip()
    model = preset.model.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Preset name cannot be empty")
    if not endpoint:
        raise HTTPException(status_code=422, detail="Endpoint cannot be empty")
    if not model:
        raise HTTPException(status_code=422, detail="Model cannot be empty")

    saved = upsert_preset(
        PRESETS_PATH,
        EndpointPreset(
            name=name,
            endpoint=endpoint,
            token=preset.token,
            model=model,
        ),
    )
    return {"preset": saved.to_dict()}

