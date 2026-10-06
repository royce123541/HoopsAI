import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, make_url, text

# Point every test at a dedicated database on the compose Postgres. This must happen before
# anything calls get_settings(), which is cached.
TEST_DATABASE_URL = os.environ.get(
    "HOOPSAI_TEST_DATABASE_URL",
    "postgresql+psycopg://hoopsai:hoopsai@localhost:5433/hoopsai_test",
)
os.environ["HOOPSAI_DATABASE_URL"] = TEST_DATABASE_URL

# psycopg's async mode needs a selector event loop; TestClient's loop follows this policy.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
DATA_TABLES = (
    "serving.predictions",
    "serving.model_versions",
    "features.game_features",
    "features.elo_ratings",
    "core.pbp_events",
    "core.team_game_stats",
    "core.games",
    "core.teams",
    "raw.api_responses",
    "raw.ingestion_runs",
)


def load_fixture(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((FIXTURES / "nba_stats" / f"{name}.json").read_text())
    return payload


def _ensure_database(url: str) -> None:
    target = make_url(url)
    admin = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target.database}
            )
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """Migrated test database. Tests using it are skipped when Postgres is unreachable."""
    try:
        _ensure_database(TEST_DATABASE_URL)
    except Exception as exc:
        pytest.skip(
            f"test database unavailable ({type(exc).__name__}); run `docker compose up -d db`"
        )
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        check=True,
        capture_output=True,
    )
    engine = create_engine(TEST_DATABASE_URL)
    yield engine
    engine.dispose()


@pytest.fixture
def db(db_engine: Engine) -> Engine:
    """Empty data tables before each test."""
    with db_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(DATA_TABLES)} CASCADE"))
    return db_engine
