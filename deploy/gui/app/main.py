"""Entrypoint for the GUI service."""

from __future__ import annotations

import logging

from .config import get_settings
from .controller import build_controller
from .interface import create_interface

LOGGER = logging.getLogger("hotmama.gui")


def launch() -> None:
    """Launch the Gradio interface."""

    settings = get_settings()
    log_format = (
        "%(message)s"
        if settings.log_json
        else "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=log_format)

    controller = build_controller(settings)
    app = create_interface(controller)
    app.queue(default_concurrency_limit=2)
    try:
        app.launch(
            server_name=settings.host,
            server_port=settings.port,
            max_threads=2,
            share=False,
            inbrowser=False,
            show_error=True,
        )
    finally:  # pragma: no cover - network cleanup
        controller.close()


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    launch()
