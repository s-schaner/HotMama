"""`hotmama serve` — run the court host."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hotmama", description="HotMama v2 court host")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the API + UI host")
    serve.add_argument("--host", default=None, help="bind address (default 0.0.0.0)")
    serve.add_argument("--port", type=int, default=None, help="port (default 8000)")
    serve.add_argument(
        "--db", type=Path, default=None, help="SQLite path (default data/hotmama.db)"
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .app import create_app
    from .config import Settings

    overrides: dict[str, object] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.db is not None:
        overrides["db_path"] = args.db
    settings = Settings(**overrides)  # type: ignore[arg-type]

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
