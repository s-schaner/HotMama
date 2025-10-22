"""Module entrypoint for running the worker."""

from __future__ import annotations

import logging

from .worker import Worker


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    worker = Worker()
    worker.run_forever()


if __name__ == "__main__":
    main()
