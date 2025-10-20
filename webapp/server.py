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
):
    data = await video.read()
    tmp_path = os.path.join(SESSION_DIR, f"upload_{video.filename}")
    with open(tmp_path, "wb") as fh:
        fh.write(data)

    try:
        frames, meta = extract_frames(tmp_path, float(fps), int(max_frames))
        images = [b64_image_data_uri(frame) for frame in frames]
        content = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in images
        )
        messages = [{"role": "user", "content": content}]

        response = call_endpoint_openai_chat(
            endpoint,
            token or None,
            model,
            messages,
            fps,
            max_pixels,
            max_tokens,
            temperature,
        )
        raw = json.dumps(response, indent=2)
        pretty = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        context = {
            "request": request,
            "pretty": pretty or "(No content)",
            "raw": raw,
            "meta": meta,
        }
        return templates.TemplateResponse("_result_panel.html", context)
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
