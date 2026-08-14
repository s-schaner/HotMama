"""`hotmama-worker` — run an analysis worker against a court host.

Examples (from any box on the network, started by its operator):

    # Stub engine — proves the pipeline
    hotmama-worker --host http://court-laptop:8000 --token $HOTMAMA_WORKER_TOKEN

    # Vision engine on a specific tier
    hotmama-worker --host http://court-laptop:8000 --token $HOTMAMA_WORKER_TOKEN \
        --engine vlm --vlm-config examples/vlm-tiers.example.json --vlm-tier deep

    # One-shot vision sanity check of a tier (no court host needed)
    hotmama-worker --probe-vlm --vlm-url http://corona:8000 --vlm-model qwen3-vl-30b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hotmama-worker")
    parser.add_argument("--host", help="court host base URL")
    parser.add_argument(
        "--token",
        default=os.environ.get("HOTMAMA_WORKER_TOKEN"),
        help="worker bearer token (or env HOTMAMA_WORKER_TOKEN)",
    )
    parser.add_argument("--name", default=socket.gethostname(), help="worker name")
    parser.add_argument(
        "--engine", default="stub", help="analysis engine: stub or vlm (default: stub)"
    )
    parser.add_argument(
        "--producer",
        default="cv_well",
        choices=["cv_well", "cv_cloud"],
        help="provenance stamped on observations",
    )
    parser.add_argument("--poll", type=float, default=5.0, help="idle poll interval (s)")
    parser.add_argument("--once", action="store_true", help="process one job and exit")

    vlm = parser.add_argument_group("vlm engine")
    vlm.add_argument("--vlm-url", help="OpenAI-compatible vision endpoint base URL")
    vlm.add_argument("--vlm-model", help="vision model name at the endpoint")
    vlm.add_argument("--vlm-key", default=os.environ.get("HOTMAMA_VLM_KEY"))
    vlm.add_argument("--vlm-frames", type=int, default=6, help="frames per clip")
    vlm.add_argument("--vlm-config", type=Path, help="JSON tier config file")
    vlm.add_argument("--vlm-tier", help="tier name from --vlm-config")
    vlm.add_argument(
        "--probe-vlm",
        action="store_true",
        help="send one synthetic image to the vision endpoint and exit",
    )
    return parser


def _resolve_vlm_options(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.vlm_url
    model = args.vlm_model
    frames = args.vlm_frames
    if args.vlm_config is not None:
        from .vlm import load_tier_config

        tiers = load_tier_config(args.vlm_config)
        tier_name = args.vlm_tier or next(iter(tiers))
        if tier_name not in tiers:
            raise SystemExit(
                f"tier {tier_name!r} not in {args.vlm_config} "
                f"(available: {', '.join(tiers)})"
            )
        tier = tiers[tier_name]
        base_url = base_url or tier.base_url
        model = model or tier.model
        frames = args.vlm_frames if args.vlm_frames != 6 else tier.max_frames
    return {
        "vlm_base_url": base_url,
        "vlm_model": model,
        "vlm_api_key": args.vlm_key,
        "vlm_frames": frames,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    vlm_options = _resolve_vlm_options(args)

    if args.probe_vlm:
        from .vlm import VlmClient, VlmError, probe

        if not vlm_options["vlm_base_url"] or not vlm_options["vlm_model"]:
            parser.error("--probe-vlm needs --vlm-url and --vlm-model (or a tier config)")
        vlm_client = VlmClient(
            base_url=str(vlm_options["vlm_base_url"]),
            model=str(vlm_options["vlm_model"]),
            api_key=vlm_options["vlm_api_key"],
        )
        try:
            result = probe(vlm_client)
        except VlmError as err:
            print(json.dumps({"error": str(err)}))
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["vision_ok"] else 2

    if not args.host:
        parser.error("--host is required to run a worker")
    if not args.token:
        parser.error("--token or HOTMAMA_WORKER_TOKEN is required")

    from .client import WorkerClient
    from .engine import make_engine

    client = WorkerClient(
        base_url=args.host,
        token=args.token,
        worker_name=args.name,
        engine=make_engine(args.engine, vlm_options),
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
