"""`hotmama-worker` — run an analysis worker against a court host.

Example (from any box on the ZeroTier network, started by its operator):

    hotmama-worker --host http://court-laptop:8000 --token $HOTMAMA_WORKER_TOKEN
"""

from __future__ import annotations

import argparse
import logging
import os
import socket


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hotmama-worker")
    parser.add_argument("--host", required=True, help="court host base URL")
    parser.add_argument(
        "--token",
        default=os.environ.get("HOTMAMA_WORKER_TOKEN"),
        help="worker bearer token (or env HOTMAMA_WORKER_TOKEN)",
    )
    parser.add_argument("--name", default=socket.gethostname(), help="worker name")
    parser.add_argument("--engine", default="stub", help="analysis engine (default: stub)")
    parser.add_argument(
        "--producer",
        default="cv_well",
        choices=["cv_well", "cv_cloud"],
        help="provenance stamped on observations",
    )
    parser.add_argument("--poll", type=float, default=5.0, help="idle poll interval (s)")
    parser.add_argument("--once", action="store_true", help="process one job and exit")
    args = parser.parse_args(argv)

    if not args.token:
        parser.error("--token or HOTMAMA_WORKER_TOKEN is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .client import WorkerClient
    from .engine import make_engine

    client = WorkerClient(
        base_url=args.host,
        token=args.token,
        worker_name=args.name,
        engine=make_engine(args.engine),
        producer=args.producer,
    )
    try:
        if args.once:
            worked = client.run_once()
            return 0 if worked else 3
        client.run_forever(poll_seconds=args.poll)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
