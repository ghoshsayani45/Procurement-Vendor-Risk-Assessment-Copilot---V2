"""Reviews carry the full request (kind, vendors, question, requester), the human decision and the trace id;
audit_events becomes append-only on SQLite too; lookup indexes.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-11
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

NEW_COLUMNS = [("request_kind", sa.Text()), ("vendor_ids", sa.JSON()), ("question", sa.Text()), ("requester", sa.Text()),
               ("requester_role", sa.Text()), ("decision", sa.Text()), ("trace_id", sa.Text()),
               ("updated_at", sa.DateTime(timezone=True))]


def upgrade() -> None:
    conn = op.get_bind()
    have = {c["name"] for c in sa.inspect(conn).get_columns("reviews")}
    for name, typ in NEW_COLUMNS:
        if name in have:
            continue
        if conn.dialect.name == "postgresql" and name == "vendor_ids":
            conn.exec_driver_sql("ALTER TABLE reviews ADD COLUMN IF NOT EXISTS vendor_ids JSONB")
        else:
            op.add_column("reviews", sa.Column(name, typ, nullable=True))
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS conversation_turns_review ON conversation_turns (review_id, id)")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS audit_events_review ON audit_events (review_id, id)")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS reviews_created ON reviews (created_at)")
    if conn.dialect.name == "sqlite":
        conn.exec_driver_sql("CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events "
                             "BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END")
        conn.exec_driver_sql("CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events "
                             "BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END")


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        conn.exec_driver_sql("DROP TRIGGER IF EXISTS audit_no_update")
        conn.exec_driver_sql("DROP TRIGGER IF EXISTS audit_no_delete")
    for name, _ in reversed(NEW_COLUMNS):
        with op.batch_alter_table("reviews") as b:
            b.drop_column(name)
