"""Persistence for the ingestion pipeline: raw payloads, idempotent core upserts, checkpoints.
All functions take a sync `Connection` so callers control the transaction boundary."""

import gzip
import json
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Connection, Table, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from hoopsai.db.models import ApiResponse, Game, IngestionRun, PbpEvent, Team, TeamGameStats
from hoopsai.sources.base import GameStatus, RawResponse

SUCCESS = "success"
FAILED = "failed"


def compress_payload(payload: dict[str, Any]) -> bytes:
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode())


def decompress_payload(blob: bytes) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(gzip.decompress(blob))
    return payload


def save_raw(conn: Connection, resp: RawResponse) -> None:
    blob = compress_payload(resp.payload)
    values = {
        "request_key": resp.request_key,
        "source": resp.source,
        "endpoint": resp.endpoint,
        "params": resp.params,
        "fetched_at": resp.fetched_at,
        "payload_gz": blob,
        "payload_bytes": len(blob),
    }
    stmt = insert(ApiResponse).values(values)
    conn.execute(
        stmt.on_conflict_do_update(
            index_elements=[ApiResponse.request_key],
            set_={k: stmt.excluded[k] for k in values if k != "request_key"},
        )
    )


def _upsert(
    conn: Connection,
    table: Table,
    rows: Sequence[Mapping[str, Any]],
    *,
    where: Any = None,
) -> int:
    """INSERT ... ON CONFLICT (pk) DO UPDATE for the columns present in the rows."""
    if not rows:
        return 0
    pk = [c.name for c in table.primary_key.columns]
    stmt = insert(table)
    update_cols: dict[str, Any] = {k: stmt.excluded[k] for k in rows[0] if k not in pk}
    if "updated_at" in table.c:
        update_cols["updated_at"] = func.now()
    conn.execute(
        stmt.on_conflict_do_update(index_elements=pk, set_=update_cols, where=where),
        list(rows),
    )
    return len(rows)


def upsert_teams(conn: Connection, rows: Sequence[Mapping[str, Any]]) -> int:
    return _upsert(conn, Team.__table__, rows)  # type: ignore[arg-type]


def upsert_games(conn: Connection, rows: Sequence[Mapping[str, Any]]) -> int:
    """Rows from different endpoints carry different columns (only the schedule knows tip
    times); each upsert touches only its own columns. A final game is never downgraded by a
    stale schedule entry."""
    table: Table = Game.__table__  # type: ignore[assignment]
    excluded = insert(table).excluded
    guard = or_(table.c.status != GameStatus.FINAL, excluded.status == GameStatus.FINAL)
    return _upsert(conn, table, rows, where=guard)


def upsert_team_game_stats(conn: Connection, rows: Sequence[Mapping[str, Any]]) -> int:
    return _upsert(conn, TeamGameStats.__table__, rows)  # type: ignore[arg-type]


def upsert_pbp_events(conn: Connection, rows: Sequence[Mapping[str, Any]]) -> int:
    return _upsert(conn, PbpEvent.__table__, rows)  # type: ignore[arg-type]


# ---------------------------------------------------------------- checkpoints


def is_done(conn: Connection, task_key: str) -> bool:
    status = conn.scalar(select(IngestionRun.status).where(IngestionRun.task_key == task_key))
    return status == SUCCESS


def mark(
    conn: Connection,
    task_key: str,
    status: str,
    *,
    rows: int | None = None,
    error: str | None = None,
) -> None:
    stmt = insert(IngestionRun).values(
        task_key=task_key, status=status, attempts=1, rows=rows, error=error
    )
    conn.execute(
        stmt.on_conflict_do_update(
            index_elements=[IngestionRun.task_key],
            set_={
                "status": status,
                "rows": rows,
                "error": error,
                "attempts": IngestionRun.attempts + 1,
                "updated_at": func.now(),
            },
        )
    )
