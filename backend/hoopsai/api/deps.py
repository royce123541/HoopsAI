from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from hoopsai.api.schemas import TeamRef
from hoopsai.db.session import get_engine


async def get_conn() -> AsyncIterator[AsyncConnection]:
    async with get_engine().connect() as conn:
        yield conn


Conn = Annotated[AsyncConnection, Depends(get_conn)]


async def load_teams(conn: AsyncConnection) -> dict[int, TeamRef]:
    rows = await conn.execute(
        text("SELECT team_id, abbreviation, city, nickname, full_name FROM core.teams")
    )
    return {r.team_id: TeamRef.model_validate(r._mapping) for r in rows}
