"""Common FastAPI dependency helpers."""

import redis
from fastapi import Request

from .parsing import JobParser


def get_redis(request: Request) -> redis.Redis:
    client = getattr(request.app.state, "redis_client", None)
    if client is None:
        raise RuntimeError("redis client not initialized")
    return client


def get_job_parser(request: Request) -> JobParser:
    parser = getattr(request.app.state, "job_parser", None)
    if parser is None:
        raise RuntimeError("job parser not initialized")
    return parser
