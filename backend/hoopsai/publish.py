"""Copy what the website and the cloud jobs need into another database (e.g. a free Neon
Postgres, 0.5 GB). The play-by-play (2.3 GB) and raw API responses (0.4 GB) stay behind:
only in-game retraining reads them, and that keeps running where they are."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import Engine, Table, create_engine, text

from hoopsai.db.models import (
    EloRating,
    Game,
    GameFeatures,
    JobRun,
    LiveWinProb,
    ModelVersionSnapshot,
    MonitoringRun,
    Prediction,
    RegisteredModel,
    Team,
    TeamGameStats,
)

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Parents before children (foreign keys).
TABLES: list[Table] = [
    m.__table__  # type: ignore[misc]
    for m in (
        Team,
        Game,
        TeamGameStats,
        EloRating,
        GameFeatures,
        ModelVersionSnapshot,
        Prediction,
        LiveWinProb,
        MonitoringRun,
        JobRun,
        RegisteredModel,
    )
]


def migrate(target_url: str) -> None:
    """Bring the target's schema to head with this package's migrations."""
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env=os.environ | {"HOOPSAI_DATABASE_URL": target_url},
        check=True,
    )


def _name(table: Table) -> str:
    return f"{table.schema}.{table.name}"


def copy_tables(source: Engine, target: Engine) -> dict[str, int]:
    """Replace the target's rows with the source's, table by table, with COPY streams."""
    with target.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(_name(t) for t in TABLES)} CASCADE"))
    counts = {}
    with source.connect() as src, target.begin() as dst:
        src_cur = src.connection.driver_connection.cursor()  # type: ignore[union-attr]
        dst_cur = dst.connection.driver_connection.cursor()  # type: ignore[union-attr]
        for table in TABLES:
            columns = ", ".join(f'"{c.name}"' for c in table.columns)
            with (
                src_cur.copy(f"COPY {_name(table)} ({columns}) TO STDOUT (FORMAT BINARY)") as out,
                dst_cur.copy(f"COPY {_name(table)} ({columns}) FROM STDIN (FORMAT BINARY)") as inp,
            ):
                for block in out:
                    inp.write(block)
            counts[_name(table)] = dst.execute(
                text(f"SELECT count(*) FROM {_name(table)}")
            ).scalar_one()
            if "id" in table.columns:  # keep identity sequences ahead of the copied ids
                dst.execute(
                    text(
                        f"SELECT setval(pg_get_serial_sequence('{_name(table)}', 'id'), "
                        f"coalesce((SELECT max(id) FROM {_name(table)}), 0) + 1, false)"
                    )
                )
            log.info("copied %s: %d rows", _name(table), counts[_name(table)])
    return counts


def publish(source: Engine, target_url: str) -> dict[str, int]:
    migrate(target_url)
    target = create_engine(target_url)
    try:
        return copy_tables(source, target)
    finally:
        target.dispose()
