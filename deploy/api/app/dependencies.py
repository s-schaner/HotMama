"""Common FastAPI dependency helpers."""

import redis
from fastapi import Request


def get_redis(request: Request) -> redis.Redis:
    client = getattr(request.app.state, "redis_client", None)
    if client is None:
        raise RuntimeError("redis client not initialized")
    return client
