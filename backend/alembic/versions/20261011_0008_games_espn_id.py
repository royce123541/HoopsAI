"""ESPN event id on core.games

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-11
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("games", sa.Column("espn_id", sa.Text(), nullable=True), schema="core")
    op.create_unique_constraint("uq_games_espn_id", "games", ["espn_id"], schema="core")


def downgrade() -> None:
    op.drop_constraint("uq_games_espn_id", "games", schema="core")
    op.drop_column("games", "espn_id", schema="core")
