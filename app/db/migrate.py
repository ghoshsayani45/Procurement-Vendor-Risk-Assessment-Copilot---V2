"""Run Alembic migrations from code (API startup, tests, docker entrypoint): `python -m app.db.migrate`."""
from __future__ import annotations

from alembic import command
from alembic.config import Config

from app.db.models import make_engine
from app.retrieval.chunking import ROOT


def upgrade(url: str | None = None, revision: str = "head") -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    engine = make_engine(url)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, revision)
    engine.dispose()


def current(url: str | None = None) -> str | None:
    from alembic.runtime.migration import MigrationContext
    engine = make_engine(url)
    with engine.connect() as conn:
        rev = MigrationContext.configure(conn).get_current_revision()
    engine.dispose()
    return rev


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    upgrade()
    print(f"database at revision {current()}")
