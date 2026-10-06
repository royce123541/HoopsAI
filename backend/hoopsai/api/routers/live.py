"""In-game win probability: full history over REST, live updates over a WebSocket fed by
Redis pub/sub (published by the `live` service and by `hoopsai replay`)."""

import asyncio
import contextlib
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from hoopsai.api.deps import Conn
from hoopsai.api.schemas import WinProbPoint, WinProbSeries
from hoopsai.db.session import get_engine
from hoopsai.live.channels import channel

router = APIRouter(tags=["live"])

# Prefer real live data; fall back to a replay of the game.
SERIES_SQL = """
    WITH chosen AS (
        SELECT source FROM serving.live_wp_snapshots WHERE game_id = :g
        GROUP BY source ORDER BY (source = 'live') DESC LIMIT 1
    )
    SELECT s.* FROM serving.live_wp_snapshots s JOIN chosen USING (source)
    WHERE s.game_id = :g ORDER BY s.action_id, s.id
"""


async def load_series(conn: AsyncConnection, game_id: str) -> WinProbSeries:
    rows = (await conn.execute(text(SERIES_SQL), {"g": game_id})).mappings().all()
    return WinProbSeries(
        game_id=game_id,
        source=rows[-1]["source"] if rows else None,
        model_version=rows[-1]["model_version"] if rows else None,
        points=[WinProbPoint.model_validate(dict(r)) for r in rows],
    )


@router.get("/games/{game_id}/winprob", response_model=WinProbSeries)
async def get_winprob(game_id: str, conn: Conn) -> WinProbSeries:
    """Every in-game win-probability point recorded for the game (empty before tip-off)."""
    return await load_series(conn, game_id)


@router.websocket("/ws/games/{game_id}")
async def winprob_stream(websocket: WebSocket, game_id: str) -> None:
    """Sends {"type": "snapshot", ...} with the history so far, then each published
    {"type": "points" | "reset", ...} message. Subscribing happens before the snapshot is
    read, so nothing published in between is lost; clients de-duplicate by action_id."""
    await websocket.accept()
    pubsub = websocket.app.state.redis.pubsub()
    await pubsub.subscribe(channel(game_id))
    try:
        async with get_engine().connect() as conn:
            series = await load_series(conn, game_id)
        await websocket.send_text(json.dumps({"type": "snapshot", **series.model_dump()}))

        async def forward() -> None:
            # A send to a client that just left raises; that simply ends the stream.
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        await websocket.send_text(message["data"])

        async def until_closed() -> None:
            with contextlib.suppress(WebSocketDisconnect):
                while True:
                    await websocket.receive_text()

        tasks = [asyncio.create_task(forward()), asyncio.create_task(until_closed())]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    finally:
        await pubsub.unsubscribe(channel(game_id))
        await pubsub.aclose()
