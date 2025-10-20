from __future__ import annotations

import csv
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


@app.post("/analyze")
async def analyze(
    request: Request,
    video: UploadFile = File(...),
    prompt: str = Form(
        "Describe rally; output JSON with rally_state, who_won, reason, timeline[{t,event,actor}]"
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

    frames, meta = extract_frames(tmp_path, float(fps), int(max_frames))
    images = [b64_image_data_uri(frame) for frame in frames]
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
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
    reply = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", str(response))
    )
    context = {
        "request": request,
        "reply": reply,
        "meta": meta,
        "video_name": video.filename,
    }
    return templates.TemplateResponse("results.html", context)


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
