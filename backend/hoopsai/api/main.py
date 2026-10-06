from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from hoopsai import __version__
from hoopsai.api.routers import games, health, model, teams
from hoopsai.config import get_settings
from hoopsai.db.session import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.redis = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await get_engine().dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="HoopsAI API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    for router in (health.router, games.router, teams.router, model.router):
        app.include_router(router, prefix="/api")
    return app


app = create_app()
