from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Iterator, List, Tuple

from fastapi import Depends, FastAPI
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.error_handlers import install_error_handlers
from app.errors import NotFoundError
from app.settings import Settings, get_settings
from app.validators import create_secure_temp_file, validate_video_upload
from core.analysis import (
    b64_image_data_uri,
    extract_frames,
    render_heatmap,
)
from llm.async_client import AsyncLLMClient
from viz.heatmap import available_renderers, render_heatmap_sexy
from viz.trajectory import (
    TrajectoryError,
    list_detection_methods,
    run_heatmap_pipeline,
)
from webapp.presets import EndpointPreset, load_presets, upsert_preset
from session.api import router as session_router

if TYPE_CHECKING:
    from session.service import SessionService

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# Initialize settings
settings = get_settings()
SESSION_PATH = settings.sessions_dir
PRESETS_PATH = SESSION_PATH / "endpoint_presets.json"
load_presets(PRESETS_PATH)


@contextlib.contextmanager
def session_service() -> Iterator["SessionService"]:
    from session.service import SessionService

    service = SessionService(settings.get_db_url(), settings.sessions_dir)
    try:
        yield service
    finally:
        service.close()


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

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
        resolved = Path(path).resolve()
        rel_path = resolved.relative_to(SESSION_PATH)
        return f"/assets/{rel_path.as_posix()}"
    except Exception:
        return None


class PresetPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    endpoint: str = Field(..., min_length=1)
    token: str = Field(default="")
    model: str = Field(..., min_length=1)


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="VolleySense",
    description="Volleyball analytics with AI-powered video analysis",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Store settings in app state
app.state.settings = settings
app.state.limiter = limiter

# Install error handlers
install_error_handlers(app)

# Add rate limiting
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS if enabled
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API routers
app.include_router(session_router)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(SESSION_PATH)), name="assets")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _coerce_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "on", "yes"}


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for monitoring and container orchestration."""
    return JSONResponse(
        {
            "status": "healthy",
            "version": "1.0.0",
            "settings": {
                "sessions_dir": str(settings.sessions_dir),
                "max_file_size_mb": settings.max_file_size_mb,
                "rate_limiting_enabled": settings.enable_rate_limiting,
            },
        }
    )


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
async def partial_sessions(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the session management dashboard panel."""
    with session_service() as service:
        sessions = service.list_sessions(limit=50)
    context = {
        "request": request,
        "sessions": sessions,
    }
    return templates.TemplateResponse("session/dashboard.html", context)


@app.post("/_api/analyze", response_class=HTMLResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def analyze_htmx(
    request: Request,
    video: Annotated[UploadFile, File(...)],
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
    current_settings: Settings = Depends(get_settings),
):
    """
    Analyze volleyball video using vision-language model.

    Validates upload, extracts frames, calls LLM API, generates heatmap.
    """
    logger.info(
        "Analysis request received",
        extra={
            "filename": video.filename,
            "model": model,
            "fps": fps,
        },
    )

    # Validate and read video file
    content = await validate_video_upload(video, current_settings)

    # Create secure temp file
    file_ext = Path(video.filename or "video.mp4").suffix
    tmp_path = create_secure_temp_file(content, file_ext, SESSION_PATH / "temp")

    try:
        frames, meta = extract_frames(str(tmp_path), float(fps), int(max_frames))
        logger.info(f"Extracted {len(frames)} frames from video")

        system_prompt = Path("prompts/qwen_volley_system.txt").read_text()
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
                + [{"type": "image_url", "image_url": {"url": url}} for url in images],
            },
        ]

        # Use async LLM client
        llm_client = AsyncLLMClient(
            endpoint=endpoint,
            model=model,
            api_key=token or None,
            timeout=current_settings.llm_timeout_seconds,
        )

        response = await llm_client.chat(
            messages,
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
            (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
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
                out_dir = SESSION_PATH / "heatmaps"
                out_dir.mkdir(parents=True, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(video.filename))[0]
                heatmap_img_path = render_heatmap(
                    valid_pts,
                    out_dir / f"hm_{base_name}.png",
                )
                csv_path = out_dir / f"hm_{base_name}.csv"
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
            img_url = _public_url(heatmap_img_path)
            if img_url:
                html += (
                    '<div class="divider">Heatmap</div>'
                    f'<img src="{img_url}" class="rounded border border-base-300">'
                )
            if heatmap_csv_path:
                csv_url = _public_url(heatmap_csv_path)
                if csv_url:
                    html += (
                        '<p class="mt-2 text-sm">Ball trajectory CSV: '
                        f'<a class="link link-primary" href="{csv_url}" target="_blank">download</a></p>'
                    )
        return HTMLResponse(html)
    except Exception as exc:
        logger.exception("Video analysis failed", exc_info=exc)
        return HTMLResponse(
            f"<div class='alert alert-error'><span>Error: {exc}</span></div>",
            status_code=500,
        )
    finally:
        # Clean up temp file
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)


@app.post("/_api/heatmap_pipeline", response_class=HTMLResponse)
async def heatmap_pipeline(
    request: Request,
    video: Annotated[UploadFile, File(...)],
    method: str = Form("auto"),
    stride: int = Form(2),
    smoothing: int = Form(3),
    min_confidence: float = Form(0.05),
    overlay: str = Form("true"),
    renderers: List[str] | None = Form(None),
) -> HTMLResponse:
    temp_path = SESSION_PATH / f"pipeline_{uuid.uuid4().hex}_{video.filename}"
    temp_path.write_bytes(await video.read())

    run_dir = SESSION_PATH / "heatmaps" / f"run_{uuid.uuid4().hex}"
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
        html = (
            f"<div class='alert alert-error'><span>Pipeline failed: {exc}</span></div>"
        )
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
    csvfile: Annotated[UploadFile, File(...)],
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

    out_path = (
        SESSION_PATH / "heatmaps" / f"heatmap_{os.path.basename(csvfile.filename)}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        saved_path = render_heatmap(points, out_path)

    return FileResponse(saved_path, media_type="image/png")


# ================================================================================
# Analytics Endpoints for Player Tracking and Stats
# ================================================================================


@app.post("/_api/analytics/process_video")
async def process_video_with_tracking(
    video: Annotated[UploadFile, File(...)],
    session_id: str = Form(...),
    enable_pose: bool = Form(True),
    enable_overlays: bool = Form(True),
    yolo_model: str = Form("yolov8n"),
    detection_confidence: float = Form(0.25),
    settings: Settings = Depends(get_settings),
):
    """
    Process video with YOLO detection, tracking, and pose estimation.

    Returns tracking results and generates overlay video.
    """
    from ingest.video_pipeline import VideoProcessingPipeline, PipelineConfig
    from viz.overlays import OverlayConfig

    # Validate video upload and capture content
    video_bytes = await validate_video_upload(video, settings)

    # Create session directory
    session_dir = SESSION_PATH / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_filename = f"{uuid.uuid4().hex}_{video.filename}"
    video_path = session_dir / "clips" / video_filename
    video_path.parent.mkdir(parents=True, exist_ok=True)

    video_path.write_bytes(video_bytes)

    # Configure pipeline
    overlay_config = OverlayConfig(
        draw_boxes=enable_overlays,
        draw_pose=enable_pose and enable_overlays,
        draw_trails=enable_overlays,
        draw_stats=False,  # Don't overlay stats on video
    )

    pipeline_config = PipelineConfig(
        yolo_model=yolo_model,
        detection_confidence=detection_confidence,
        enable_pose=enable_pose,
        enable_overlays=enable_overlays,
        overlay_config=overlay_config,
        enable_heatmap=True,
    )

    # Initialize pipeline
    pipeline = VideoProcessingPipeline(config=pipeline_config)

    # Output path for video with overlays
    output_path = session_dir / "processed" / f"tracked_{video_filename}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process video
    try:
        result = pipeline.process_video(
            str(video_path),
            output_path=str(output_path) if enable_overlays else None,
        )

        # Get player analytics
        analytics = pipeline.get_player_analytics(result)

        # Save heatmap if generated
        heatmap_path = None
        if result.heatmap is not None:
            heatmap_path = session_dir / "heatmaps" / f"heatmap_{video_filename}.png"
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            import cv2

            cv2.imwrite(str(heatmap_path), result.heatmap)

        return JSONResponse(
            {
                "success": True,
                "video_path": _public_url(video_path),
                "output_path": _public_url(output_path) if enable_overlays else None,
                "heatmap_path": _public_url(heatmap_path) if heatmap_path else None,
                "total_frames": result.total_frames,
                "processed_frames": result.processed_frames,
                "total_tracks": len(result.track_histories),
                "analytics": analytics,
            }
        )

    except Exception as e:
        logger.error(f"Video processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Video processing failed: {str(e)}"
        )


@app.get("/_api/analytics/stats")
async def get_analytics_stats(
    session_id: str | None = None,
):
    """
    Get comprehensive player statistics.

    If session_id is provided, returns stats for that session.
    Otherwise returns aggregate stats across all sessions.
    """
    if session_id:
        try:
            with session_service() as svc:
                svc.load_session(session_id)
                rollup = svc.get_rollup(session_id)
        except NotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

        players = [
            {
                "track_id": index,
                "team": entry.team_side or "",
                "number": entry.number or "",
                "event_type": entry.event_type,
                "count": entry.count,
            }
            for index, entry in enumerate(rollup, start=1)
        ]

        return JSONResponse({"players": players})
    else:
        # Return empty for now - could implement cross-session analytics
        return JSONResponse({"players": []})


@app.post("/_api/analytics/export")
async def export_analytics(
    session_id: str = Form(...),
):
    """
    Export analytics data as CSV.
    """
    export_dir = settings.sessions_dir / session_id / "exports"

    with session_service() as svc:
        export_result = svc.export_csv(session_id, str(export_dir))

    events_csv = export_result.get("events")
    if events_csv and Path(events_csv).exists():
        return FileResponse(
            events_csv,
            media_type="text/csv",
            filename=f"stats_{session_id}.csv",
        )

    raise HTTPException(status_code=404, detail="No analytics data found")


@app.get("/_partial/analytics")
async def analytics_dashboard_partial(request: Request):
    """Render analytics dashboard partial."""
    return templates.TemplateResponse("_analytics_dashboard.html", {"request": request})


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
