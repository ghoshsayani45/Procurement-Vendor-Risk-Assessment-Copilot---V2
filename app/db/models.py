"""SQLAlchemy Core tables for the API's own state: reviews, conversation turns, append-only audit events.

The same definitions work on SQLite (default, one file shared with the document store) and Postgres (`make up`).
The schema itself is created and changed only by Alembic migrations (migrations/versions/), never by create_all().
"""
from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from app.retrieval.chunking import ROOT

metadata = sa.MetaData()
JSONType = sa.JSON().with_variant(JSONB(), "postgresql")

reviews = sa.Table(
    "reviews", metadata,
    sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
    sa.Column("vendor_id", sa.Text, nullable=False),
    sa.Column("review_type", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="OPEN"),
    sa.Column("as_of", sa.Date, nullable=False),
    sa.Column("score_json", JSONType),
    sa.Column("report_json", JSONType),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    # added in migration 0002
    sa.Column("request_kind", sa.Text),
    sa.Column("vendor_ids", JSONType),
    sa.Column("question", sa.Text),
    sa.Column("requester", sa.Text),
    sa.Column("requester_role", sa.Text),
    sa.Column("decision", sa.Text),
    sa.Column("trace_id", sa.Text),
    sa.Column("updated_at", sa.DateTime(timezone=True)),
)

conversation_turns = sa.Table(
    "conversation_turns", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True, autoincrement=True),
    sa.Column("review_id", sa.Uuid(as_uuid=False), nullable=False),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("content", JSONType, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
)

audit_events = sa.Table(
    "audit_events", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True, autoincrement=True),
    sa.Column("review_id", sa.Uuid(as_uuid=False)),
    sa.Column("actor", sa.Text, nullable=False),
    sa.Column("action", sa.Text, nullable=False),
    sa.Column("decision", sa.Text),
    sa.Column("detail", JSONType),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
)


# Ask the Copilot (D45, migration 0003): chat sessions are not tied to a review
chat_sessions = sa.Table(
    "chat_sessions", metadata,
    sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
    sa.Column("owner", sa.Text, nullable=False),
    sa.Column("title", sa.Text),
    sa.Column("state", JSONType),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True)),
)

chat_messages = sa.Table(
    "chat_messages", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True, autoincrement=True),
    sa.Column("session_id", sa.Uuid(as_uuid=False), nullable=False),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("content", JSONType, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
)


def database_url() -> str:
    """SQLAlchemy URL: Postgres from DATABASE_URL when STORE_BACKEND=postgres, else the SQLite store file."""
    if os.getenv("STORE_BACKEND", "sqlite").strip().lower() in ("postgres", "pg"):
        url = os.getenv("DATABASE_URL") or (
            f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.environ['POSTGRES_DB']}")
        return url.replace("postgresql://", "postgresql+psycopg://", 1) if url.startswith("postgresql://") else url
    path = Path(os.getenv("STORE_SQLITE_PATH", str(ROOT / "data" / "processed" / "store.sqlite")))
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def make_engine(url: str | None = None):
    url = url or database_url()
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    eng = sa.create_engine(url, future=True, **kw)
    if url.startswith("sqlite"):
        @sa.event.listens_for(eng, "connect")
        def _fk(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys = ON")
    return eng
