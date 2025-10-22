#!/usr/bin/env python3
"""Simple utility to block until Redis is reachable."""

from __future__ import annotations

import os
import sys
import time

import redis


def main() -> int:
    url = os.environ.get("REDIS_URL", "redis://queue:6379/0")
    timeout = int(os.environ.get("WAIT_FOR_REDIS_TIMEOUT", "30"))
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            client = redis.from_url(url)
            client.ping()
            return 0
        except redis.RedisError:
            time.sleep(1)
    print(f"Timed out waiting for redis at {url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
