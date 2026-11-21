"""Alembic environment: URL from app.db.models.database_url() (SQLite store file, or Postgres with STORE_BACKEND=postgres)."""
from alembic import context
from dotenv import load_dotenv

load_dotenv()
from app.db.models import database_url, make_engine, metadata  # noqa: E402

config = context.config
target_metadata = metadata


def run_migrations_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url") or database_url(), target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = config.attributes.get("connection")
    if connectable is not None:                      # called from app.db.migrate with an open connection
        context.configure(connection=connectable, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = make_engine(config.get_main_option("sqlalchemy.url") or None)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
