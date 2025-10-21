"""Centralized error handling for the application."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Structured error response model."""

    error: str = Field(..., description="Error type or name")
    detail: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Machine-readable error code")
    status_code: int = Field(..., description="HTTP status code")


class ValidationError(HTTPException):
    """Raised when input validation fails."""

    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


class FileTooLargeError(HTTPException):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, detail: str = "File size exceeds maximum allowed"):
        super().__init__(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=detail)


class InvalidFileTypeError(HTTPException):
    """Raised when uploaded file type is not allowed."""

    def __init__(self, detail: str = "File type not allowed"):
        super().__init__(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=detail)


class LLMAPIError(HTTPException):
    """Raised when LLM API call fails."""

    def __init__(self, detail: str = "LLM API request failed"):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


class VideoProcessingError(HTTPException):
    """Raised when video processing fails."""

    def __init__(self, detail: str = "Video processing failed"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


def error_to_response(error: str, detail: str, code: str, status_code: int) -> ErrorResponse:
    """Create a structured error response."""
    return ErrorResponse(
        error=error,
        detail=detail,
        code=code,
        status_code=status_code,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse | HTMLResponse:
    """Handle HTTPException with structured response."""
    error_response = error_to_response(
        error=exc.__class__.__name__,
        detail=str(exc.detail),
        code=_get_error_code(exc),
        status_code=exc.status_code,
    )

    # Return HTML for HTMX requests, JSON otherwise
    if "HX-Request" in request.headers:
        html = f'<div class="alert alert-error"><span>{error_response.detail}</span></div>'
        return HTMLResponse(content=html, status_code=exc.status_code)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse | HTMLResponse:
    """Handle general exceptions with structured response."""
    logger.exception("Unhandled exception occurred", exc_info=exc)

    error_response = error_to_response(
        error=exc.__class__.__name__,
        detail="An internal error occurred",
        code="INTERNAL_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )

    # Include detailed error in debug mode
    if hasattr(request.app.state, "settings") and request.app.state.settings.debug:
        error_response.detail = str(exc)

    # Return HTML for HTMX requests, JSON otherwise
    if "HX-Request" in request.headers:
        html = f'<div class="alert alert-error"><span>{error_response.detail}</span></div>'
        return HTMLResponse(content=html, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


def _get_error_code(exc: HTTPException) -> str:
    """Extract error code from exception."""
    error_codes = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }
    return error_codes.get(exc.status_code, "UNKNOWN_ERROR")


def install_error_handlers(app: Any) -> None:
    """Install error handlers on the FastAPI app."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
