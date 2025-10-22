"""Entrypoint for the API service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .dependencies import get_redis
from .parsing import create_job_parser
from .routes import router

LOGGER = logging.getLogger("hotmama.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log_format = (
        "%(message)s"
        if settings.log_json
        else "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=log_format)
    LOGGER.info("starting api service", extra={"service": settings.service_name})
    client = redis.from_url(settings.redis_url, decode_responses=False)
    parser = create_job_parser(settings)
    try:
        app.state.redis_client = client
        app.state.job_parser = parser
        yield
    finally:
        client.close()
        if hasattr(parser, "close"):
            parser.close()  # type: ignore[attr-defined]
        LOGGER.info("stopped api service", extra={"service": settings.service_name})


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title="HotMama API", lifespan=lifespan)

    if settings.allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.include_router(router, prefix=settings.api_prefix)
    application.dependency_overrides[redis.Redis] = get_redis
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
