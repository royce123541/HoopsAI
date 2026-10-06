import subprocess
import sys
from pathlib import Path

from hoopsai.db.base import SCHEMAS

BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_offline_migration_creates_all_schemas() -> None:
    """Render migrations as SQL (no DB needed); every data-layer schema must be created."""
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    for schema in SCHEMAS:
        assert f"CREATE SCHEMA IF NOT EXISTS {schema}" in result.stdout
