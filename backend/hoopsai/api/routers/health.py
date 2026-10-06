import asyncio
from collections.abc import Awaitable
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import text

from hoopsai import __version__
from hoopsai.db.session import get_engine

router = APIRouter(tags=["health"])

CHECK_TIMEOUT_S = 2.0


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, str]


async def _check_database() -> None:
    async with get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))


async def _check_redis(request: Request) -> None:
    await request.app.state.redis.ping()


async def _run(check: Awaitable[None]) -> str:
    try:
        await asyncio.wait_for(check, CHECK_TIMEOUT_S)
    except Exception as exc:  # report any dependency failure instead of raising
        return f"error: {type(exc).__name__}"
    return "ok"


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness plus dependency status. Always 200 so the API stays up while deps recover."""
    db, cache = await asyncio.gather(_run(_check_database()), _run(_check_redis(request)))
    checks = {"database": db, "redis": cache}
    status: Literal["ok", "degraded"] = (
        "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    )
    return HealthResponse(status=status, version=__version__, checks=checks)
