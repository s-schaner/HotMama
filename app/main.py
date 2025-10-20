from __future__ import annotations

import argparse
import logging
import os

from app.config import load_config
from app.logging import configure_logging
from webapp.server import run as run_server


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="VolleySense FastAPI App")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    args = parser.parse_args()

    config = load_config()
    configure_logging(config.sessions_dir / "logs")
    os.environ.setdefault("VOLLEYSENSE_SESSIONS", str(config.sessions_dir))
    LOGGER.info(
        "Starting VolleySense FastAPI server",
        extra={"host": args.host, "port": args.port, "sessions_dir": str(config.sessions_dir)},
    )

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
