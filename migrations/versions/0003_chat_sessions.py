"""Ask the Copilot chat sessions and messages (D45).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-21
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

POSTGRES = [
    """CREATE TABLE IF NOT EXISTS chat_sessions (id UUID PRIMARY KEY, owner TEXT NOT NULL, title TEXT, state JSONB,
       created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ)""",
    """CREATE TABLE IF NOT EXISTS chat_messages (id BIGSERIAL PRIMARY KEY,
       session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
       role TEXT NOT NULL CHECK (role IN ('user','assistant')), content JSONB NOT NULL,
       created_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
]
SQLITE = [
    """CREATE TABLE IF NOT EXISTS chat_sessions (id CHAR(32) PRIMARY KEY, owner TEXT NOT NULL, title TEXT, state JSON,
       created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME)""",
    """CREATE TABLE IF NOT EXISTS chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
       session_id CHAR(32) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
       role TEXT NOT NULL CHECK (role IN ('user','assistant')), content JSON NOT NULL,
       created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
]
INDEXES = ["CREATE INDEX IF NOT EXISTS chat_messages_session ON chat_messages (session_id, id)",
           "CREATE INDEX IF NOT EXISTS chat_sessions_owner ON chat_sessions (owner, updated_at)"]


def upgrade() -> None:
    conn = op.get_bind()
    for stmt in (POSTGRES if conn.dialect.name == "postgresql" else SQLITE) + INDEXES:
        conn.exec_driver_sql(stmt)


def downgrade() -> None:
    conn = op.get_bind()
    conn.exec_driver_sql("DROP TABLE IF EXISTS chat_messages")
    conn.exec_driver_sql("DROP TABLE IF EXISTS chat_sessions")
