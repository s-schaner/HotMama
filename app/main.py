from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.bus import EventBus
from app.config import load_config
from app.logging import configure_logging
from session.service import SessionService
from ui.gradio_app import GradioVolleyApp


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="VolleySense Gradio App")
    parser.add_argument("--share", action="store_true", help="Launch Gradio with share mode")
    parser.add_argument("--auth", nargs=2, metavar=("USER", "PASS"), help="Optional basic auth")
    args = parser.parse_args()

    config = load_config()
    configure_logging(config.sessions_dir / "logs")
    LOGGER.info("Starting VolleySense", extra={"db_url": config.db_url})

    service = SessionService(config.db_url, config.sessions_dir)
    bus = EventBus()
    app = GradioVolleyApp(service, config.sessions_dir, Path("plugins"), bus)
    auth_tuple = tuple(args.auth) if args.auth else None
    app.launch(share=args.share, auth=auth_tuple)


if __name__ == "__main__":
    main()
