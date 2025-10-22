"""Input validation utilities."""
import secrets
import tempfile
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile

from app.error_handlers import FileTooLargeError, InvalidFileTypeError
from app.settings import Settings


async def validate_video_upload(
    video: UploadFile,
    settings: Settings,
) -> bytes:
    """
    Validate uploaded video file.

    Args:
        video: The uploaded file
        settings: Application settings

    Returns:
        Video file content as bytes

    Raises:
        InvalidFileTypeError: If file type is not allowed
        FileTooLargeError: If file size exceeds limit
    """
    # Validate file extension
    if not video.filename:
        raise InvalidFileTypeError("Filename is required")

    ext = Path(video.filename).suffix.lower()
    if ext not in settings.allowed_video_extensions:
        raise InvalidFileTypeError(
            f"File type '{ext}' not allowed. Allowed types: {', '.join(settings.allowed_video_extensions)}"
        )

    # Read and validate file size
    content = await video.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > settings.max_file_size_mb:
        raise FileTooLargeError(
            f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed ({settings.max_file_size_mb}MB)"
        )

    return content


def create_secure_temp_file(
    content: bytes,
    suffix: str,
    directory: Path,
) -> Path:
    """
    Create a secure temporary file with random name.

    Args:
        content: File content to write
        suffix: File extension (e.g., '.mp4')
        directory: Directory to create temp file in

    Returns:
        Path to the created temporary file
    """
    directory.mkdir(parents=True, exist_ok=True)

    # Generate secure random filename
    random_name = secrets.token_hex(16)
    temp_path = directory / f"tmp_{random_name}{suffix}"

    # Write content
    temp_path.write_bytes(content)

    return temp_path


def validate_coordinate(x: float, y: float, x_max: float = 9.0, y_max: float = 18.0) -> bool:
    """
    Validate court coordinate is within bounds.

    Args:
        x: X coordinate (court width)
        y: Y coordinate (court length)
        x_max: Maximum X value
        y_max: Maximum Y value

    Returns:
        True if coordinate is valid
    """
    return 0 <= x <= x_max and 0 <= y <= y_max


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing dangerous characters.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = Path(filename).name

    # Remove dangerous characters
    dangerous_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*", "\0"]
    for char in dangerous_chars:
        filename = filename.replace(char, "_")

    # Limit length
    if len(filename) > max_length:
        stem = Path(filename).stem[:max_length - 10]
        suffix = Path(filename).suffix
        filename = f"{stem}{suffix}"

    return filename or "unnamed"
