from __future__ import annotations

import csv
import json
import os
from typing import List, Tuple

from fastapi import FastAPI
from fastapi import File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.analysis import (
    b64_image_data_uri,
    call_endpoint_openai_chat,
    extract_frames,
    render_heatmap,
)


SESSION_DIR = os.environ.get(
    "VOLLEYSENSE_SESSIONS", os.path.join(os.getcwd(), "sessions")
)
os.makedirs(SESSION_DIR, exist_ok=True)

app = FastAPI(title="VolleySense")
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/_partial/heatmap", response_class=HTMLResponse)
async def partial_heatmap(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_heatmap_panel.html", {"request": request})


@app.get("/_partial/sessions", response_class=HTMLResponse)
async def partial_sessions(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_sessions_panel.html", {"request": request})


@app.post("/_api/analyze", response_class=HTMLResponse)
async def analyze_htmx(
    request: Request,
    video: UploadFile = File(...),
    prompt: str = Form(
        "Describe rally; output STRICT JSON with:\n"
        '{"rally_state": "...", "who_won": "teamA|teamB|null", "reason":"...", '
        '"timeline":[{"t": 12.3, "event":"attack", "actor":"teamA#12"}]}'
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
):
    data = await video.read()
    tmp_path = os.path.join(SESSION_DIR, f"upload_{video.filename}")
    with open(tmp_path, "wb") as fh:
        fh.write(data)

    try:
        frames, meta = extract_frames(tmp_path, float(fps), int(max_frames))
        system_prompt = open("prompts/qwen_volley_system.txt").read()
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
                heatmap_img_path = render_heatmap(
                    valid_pts,
                    os.path.join(
                        out_dir, f"hm_{os.path.basename(video.filename)}.png"
                    ),
                )

        context = {
            "request": request,
            "pretty": pretty or "(No content)",
            "raw": raw,
            "meta": meta,
        }
        html = templates.get_template("_result_panel.html").render(context)
        if heatmap_img_path:
            html += (
                "<div class=\"divider\">Heatmap</div>"
                f"<img src=\"/static/..{heatmap_img_path.replace(os.getcwd(), '')}\" "
                "class=\"rounded border border-base-300\">"
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


@app.post("/heatmap")
async def heatmap(csvfile: UploadFile = File(...)) -> FileResponse:
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
    saved_path = render_heatmap(points, out_path)
    return FileResponse(saved_path, media_type="image/png")


def run(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run("webapp.server:app", host=host, port=port, reload=False)
