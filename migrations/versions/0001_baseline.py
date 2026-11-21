"""Baseline: every table of the Week 2 schema.

Postgres: runs app/db/schema.sql (the same file docker compose loads on first start; every statement is idempotent).
SQLite: the document-store tables from app/db/store.py plus the API tables (reviews, conversation_turns, audit_events).

Revision ID: 0001
Revises:
Create Date: 2026-11-11
"""
import re
from pathlib import Path

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

ROOT = Path(__file__).resolve().parents[2]

SQLITE_API = """
CREATE TABLE IF NOT EXISTS financial_metrics (vendor_id TEXT NOT NULL REFERENCES vendors(id), period TEXT NOT NULL,
    metric TEXT NOT NULL, value NUMERIC NOT NULL, source TEXT NOT NULL, PRIMARY KEY (vendor_id, period, metric));
CREATE TABLE IF NOT EXISTS reviews (id CHAR(32) PRIMARY KEY, vendor_id TEXT NOT NULL REFERENCES vendors(id),
    review_type TEXT NOT NULL CHECK (review_type IN ('NEW_VENDOR','RENEWAL')), status TEXT NOT NULL DEFAULT 'OPEN',
    as_of DATE NOT NULL, score_json JSON, report_json JSON, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS conversation_turns (id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id CHAR(32) NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','tool')), content JSON NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS audit_events (id INTEGER PRIMARY KEY AUTOINCREMENT, review_id CHAR(32) REFERENCES reviews(id),
    actor TEXT NOT NULL, action TEXT NOT NULL, decision TEXT, detail JSON,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""


def _statements(sql: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in sql.split(";") if s.strip()]


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        for stmt in _statements((ROOT / "app" / "db" / "schema.sql").read_text(encoding="utf-8")):
            conn.exec_driver_sql(stmt)
    else:
        from app.db.store import SQLITE_SCHEMA
        for stmt in _statements(SQLITE_SCHEMA + SQLITE_API):
            conn.exec_driver_sql(stmt)


def downgrade() -> None:
    raise NotImplementedError("The baseline is not reversible: drop the database volume instead (make down, then remove pvrc_pgdata).")
