"""
FastAPI router for session management CRUD operations.

This module provides RESTful endpoints for creating, reading, updating,
and deleting volleyball analysis sessions.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.errors import NotFoundError
from app.settings import get_settings, Settings
from session import schemas
from session.service import SessionService

logger = logging.getLogger(__name__)

# Initialize templates
templates = Jinja2Templates(directory="webapp/templates")

# Create API router
router = APIRouter(prefix="/_api/sessions", tags=["sessions"])


def get_session_service(settings: Settings = Depends(get_settings)) -> SessionService:
    """Dependency to get SessionService instance."""
    db_url = settings.db_url or f"sqlite:///{settings.sessions_dir / 'volleysense.db'}"
    return SessionService(db_url, settings.sessions_dir)


class SessionUpdateRequest(BaseModel):
    """Request model for updating session metadata."""

    title: Optional[str] = None
    venue: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class SessionDetailsResponse(BaseModel):
    """Detailed session information including clips and stats."""

    session: schemas.SessionOut
    clips: List[schemas.ClipOut]
    teams: Dict[str, Optional[str]]
    rosters: Dict[str, List[schemas.RosterEntry]]
    event_count: int
    rollup_summary: Dict[str, Any]


class BulkDeleteRequest(BaseModel):
    """Request model for bulk session deletion."""

    session_ids: List[str] = Field(..., min_length=1)


# ============================================================================
# Session CRUD Endpoints
# ============================================================================


@router.post("/", response_model=dict)
async def create_session(
    request: Request,
    service: SessionService = Depends(get_session_service),
) -> dict:
    """
    Create a new session.

    Accepts both JSON and form data.

    Args:
        request: FastAPI request
        service: SessionService instance

    Returns:
        Dictionary with session_id and success message
    """
    try:
        # Try to parse as JSON first, then fall back to form data
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            data_dict = await request.json()
        else:
            # Form data
            form = await request.form()
            data_dict = {
                "title": form.get("title"),
                "venue": form.get("venue"),
                "meta": {},
            }

            # Add any additional form fields to meta
            if form.get("notes"):
                data_dict["meta"]["notes"] = form.get("notes")
            if form.get("match_date"):
                data_dict["meta"]["match_date"] = form.get("match_date")

        data = schemas.SessionCreate(**data_dict)
        session_id = service.create_session(data)
        logger.info(f"Created session: {session_id} - {data.title}")
        return {
            "session_id": session_id,
            "message": "Session created successfully",
        }
    except Exception as exc:
        logger.error(f"Failed to create session: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create session: {exc}")


@router.get("/", response_model=List[schemas.SessionOut])
async def list_sessions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    service: SessionService = Depends(get_session_service),
) -> List[schemas.SessionOut]:
    """
    List all sessions with pagination and optional search.

    Args:
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip
        search: Optional search term for title/venue
        service: SessionService instance

    Returns:
        List of session objects
    """
    try:
        sessions = service.list_sessions(limit=limit, offset=offset, search=search)
        return sessions
    except Exception as exc:
        logger.error(f"Failed to list sessions: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {exc}")


@router.get("/{session_id}", response_model=SessionDetailsResponse)
async def get_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> SessionDetailsResponse:
    """
    Get detailed information about a specific session.

    Args:
        session_id: UUID of the session
        service: SessionService instance

    Returns:
        Detailed session information including clips and stats
    """
    try:
        session_data = service.load_session(session_id)
        return SessionDetailsResponse(**session_data)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        logger.error(f"Failed to load session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load session: {exc}")


@router.patch("/{session_id}", response_model=dict)
async def update_session(
    request: Request,
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> dict:
    """
    Update session metadata.

    Accepts both JSON and form data.

    Args:
        request: FastAPI request
        session_id: UUID of the session
        service: SessionService instance

    Returns:
        Success message
    """
    try:
        # Try to parse as JSON first, then fall back to form data
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            updates = await request.json()
        else:
            # Form data
            form = await request.form()
            updates = {}

            if form.get("title"):
                updates["title"] = form.get("title")
            if form.get("venue") is not None:
                updates["venue"] = form.get("venue")

        service.update_session(session_id, updates)
        logger.info(f"Updated session: {session_id}")
        return {"message": "Session updated successfully"}
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        logger.error(f"Failed to update session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update session: {exc}")


@router.delete("/{session_id}", response_model=dict)
async def delete_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> dict:
    """
    Delete a session and all associated data.

    Args:
        session_id: UUID of the session
        service: SessionService instance

    Returns:
        Success message
    """
    try:
        service.delete_session(session_id)
        logger.info(f"Deleted session: {session_id}")
        return {"message": "Session deleted successfully"}
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        logger.error(f"Failed to delete session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {exc}")


@router.post("/bulk-delete", response_model=dict)
async def bulk_delete_sessions(
    data: BulkDeleteRequest,
    service: SessionService = Depends(get_session_service),
) -> dict:
    """
    Delete multiple sessions in bulk.

    Args:
        data: List of session IDs to delete
        service: SessionService instance

    Returns:
        Summary of deletion results
    """
    deleted = []
    errors = []

    for session_id in data.session_ids:
        try:
            service.delete_session(session_id)
            deleted.append(session_id)
        except Exception as exc:
            errors.append({"session_id": session_id, "error": str(exc)})

    logger.info(f"Bulk delete: {len(deleted)} deleted, {len(errors)} errors")
    return {
        "deleted_count": len(deleted),
        "deleted_ids": deleted,
        "errors": errors,
    }


# ============================================================================
# Artifacts & Exports
# ============================================================================


@router.post("/{session_id}/artifacts", response_model=dict)
async def attach_artifact(
    session_id: str,
    artifact_type: str,
    file_path: str,
    metadata: Dict = {},
    service: SessionService = Depends(get_session_service),
) -> dict:
    """
    Attach an artifact (heatmap, export, etc.) to a session.

    Args:
        session_id: UUID of the session
        artifact_type: Type of artifact (e.g., 'heatmap', 'export')
        file_path: Path to the artifact file
        metadata: Additional metadata
        service: SessionService instance

    Returns:
        Success message with artifact ID
    """
    try:
        artifact_id = service.attach_artifact(
            session_id=session_id,
            artifact_type=artifact_type,
            file_path=file_path,
            metadata=metadata,
        )
        return {
            "artifact_id": artifact_id,
            "message": "Artifact attached successfully",
        }
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        logger.error(f"Failed to attach artifact: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to attach artifact: {exc}")


@router.post("/{session_id}/export", response_model=dict)
async def export_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Export session data to CSV files.

    Args:
        session_id: UUID of the session
        service: SessionService instance
        settings: Application settings

    Returns:
        Paths to exported CSV files
    """
    try:
        export_dir = settings.sessions_dir / session_id / "exports"
        result = service.export_csv(session_id, str(export_dir))
        logger.info(f"Exported session {session_id} to {export_dir}")
        return {
            "events_csv": result["events"],
            "rollup_csv": result["rollup"],
        }
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        logger.error(f"Failed to export session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export session: {exc}")


# ============================================================================
# UI Partials (HTMX endpoints)
# ============================================================================


@router.get("/ui/dashboard", response_class=HTMLResponse)
async def session_dashboard(
    request: Request,
    service: SessionService = Depends(get_session_service),
) -> HTMLResponse:
    """
    Render the session dashboard UI partial.

    Returns:
        HTML response with session list and management UI
    """
    try:
        sessions = service.list_sessions(limit=50)
        context = {
            "request": request,
            "sessions": sessions,
        }
        return templates.TemplateResponse("session/dashboard.html", context)
    except Exception as exc:
        logger.error(f"Failed to render dashboard: {exc}", exc_info=True)
        error_html = f'<div class="alert alert-error"><span>Failed to load sessions: {exc}</span></div>'
        return HTMLResponse(error_html, status_code=500)


@router.get("/ui/editor/{session_id}", response_class=HTMLResponse)
async def session_editor(
    request: Request,
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> HTMLResponse:
    """
    Render the session editor UI partial.

    Args:
        request: FastAPI request
        session_id: UUID of the session to edit
        service: SessionService instance

    Returns:
        HTML response with session editor form
    """
    try:
        session_data = service.load_session(session_id)
        context = {
            "request": request,
            "session": session_data["session"],
            "clips": session_data.get("clips", []),
            "teams": session_data.get("teams", {}),
            "rosters": session_data.get("rosters", {}),
            "rollup_summary": session_data.get("rollup_summary", {}),
        }
        return templates.TemplateResponse("session/editor.html", context)
    except NotFoundError:
        error_html = f'<div class="alert alert-error"><span>Session {session_id} not found</span></div>'
        return HTMLResponse(error_html, status_code=404)
    except Exception as exc:
        logger.error(f"Failed to render editor for {session_id}: {exc}", exc_info=True)
        error_html = f'<div class="alert alert-error"><span>Failed to load session: {exc}</span></div>'
        return HTMLResponse(error_html, status_code=500)


@router.get("/ui/create", response_class=HTMLResponse)
async def session_create_form(request: Request) -> HTMLResponse:
    """
    Render the session creation form.

    Returns:
        HTML response with session creation form
    """
    context = {"request": request}
    return templates.TemplateResponse("session/_create_form.html", context)
