from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from tools.dependencies import format_environment_summary, resolve_profile


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir = _ensure_directory(log_dir)
    log_path = log_dir / "container.log"
    logger = logging.getLogger("volleysense.docker")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Docker runtime log started: %s", datetime.utcnow().isoformat() + "Z")
    logger.info("Logs written to %s", log_path)
    return logger


def _public_host(host: str) -> str:
    return "localhost" if host in {"0.0.0.0", "::"} else host


def _launch_uvicorn(host: str, port: int, env: dict, logger: logging.Logger) -> int:
    command: List[str] = [
        sys.executable,
        "-m",
        "uvicorn",
        "webapp.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    logger.info("Starting uvicorn: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    def _forward_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            logger.info("[uvicorn] %s", line.rstrip())

    try:
        _forward_output()
    except KeyboardInterrupt:
        logger.warning("Received interrupt; stopping server...")
        process.send_signal(signal.SIGTERM)
    finally:
        return_code = process.wait()
        logger.info("uvicorn exited with code %s", return_code)
    return return_code


def main() -> int:
    sessions_dir = Path(os.environ.get("VOLLEYSENSE_SESSIONS", "/app/sessions"))
    logs_dir = Path(os.environ.get("VOLLEYSENSE_LOG_DIR", "/app/logs"))
    _ensure_directory(sessions_dir)
    logger = _setup_logging(logs_dir)

    profile_key = os.environ.get("VOLLEYSENSE_PROFILE", "auto")
    profile, hardware = resolve_profile(profile_key)
    summary_lines = format_environment_summary(profile, hardware)

    host = os.environ.get("VOLLEYSENSE_HOST", "0.0.0.0")
    port = int(os.environ.get("VOLLEYSENSE_PORT", "8000"))
    public_host = _public_host(host)
    gui_url = f"http://{public_host}:{port}/"

    summary = {
        "profile": profile.key,
        "host": host,
        "port": port,
        "public_url": gui_url,
        "hardware": summary_lines,
    }

    logger.info("Container runtime summary:\n%s", json.dumps(summary, indent=2))
    logger.info("GUI available at %s", gui_url)

    env = os.environ.copy()
    env.setdefault("VOLLEYSENSE_SESSIONS", str(sessions_dir))

    exit_code = _launch_uvicorn(host, port, env, logger)
    if exit_code != 0:
        logger.error("VolleySense server terminated unexpectedly (exit code %s)", exit_code)
        crash_log = logs_dir / "container-crash.log"
        crash_log.write_text(
            f"Server crashed at {datetime.utcnow().isoformat()}Z with exit code {exit_code}\n",
            encoding="utf-8",
        )
        logger.error("Crash details recorded in %s", crash_log)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
