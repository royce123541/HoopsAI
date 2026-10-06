from logging.config import fileConfig

from sqlalchemy import create_engine, pool

import hoopsai.db.models  # noqa: F401  (registers tables on Base.metadata)
from alembic import context
from hoopsai.config import get_settings
from hoopsai.db.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
DATABASE_URL = get_settings().database_url

# Tables live in the raw/core/features/serving schemas, so autogenerate must look beyond `public`.
CONFIGURE_OPTS = {"target_metadata": target_metadata, "include_schemas": True}


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a database connection (`alembic upgrade head --sql`)."""
    context.configure(url=DATABASE_URL, literal_binds=True, **CONFIGURE_OPTS)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(DATABASE_URL, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, **CONFIGURE_OPTS)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
