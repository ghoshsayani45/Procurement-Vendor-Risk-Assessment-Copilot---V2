"""Persistence for reviews, conversation turns and audit events. SQLAlchemy Core on SQLite or Postgres."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import sqlalchemy as sa

from app.db.models import audit_events, conversation_turns, make_engine, reviews


def _now():
    return datetime.now(timezone.utc)


def _row(r) -> dict:
    d = dict(r._mapping)
    for k, v in d.items():
        if isinstance(v, (datetime, date)):
            d[k] = v.isoformat()
    return d


class Repository:
    def __init__(self, url: str | None = None, engine=None):
        self.engine = engine or make_engine(url)

    # ---- reviews
    def create_review(self, *, vendor_ids: list[str], review_type: str, request_kind: str, as_of: date, question: str | None,
                      requester: str, requester_role: str, trace_id: str | None) -> str:
        rid = str(uuid.uuid4())
        with self.engine.begin() as c:
            c.execute(reviews.insert().values(id=rid, vendor_id=vendor_ids[0], review_type=review_type, status="OPEN", as_of=as_of,
                                              request_kind=request_kind, vendor_ids=vendor_ids, question=question,
                                              requester=requester, requester_role=requester_role, trace_id=trace_id,
                                              created_at=_now(), updated_at=_now()))
        return rid

    def update_review(self, rid: str, **values) -> None:
        with self.engine.begin() as c:
            c.execute(reviews.update().where(reviews.c.id == rid).values(**values, updated_at=_now()))

    def get_review(self, rid: str) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(sa.select(reviews).where(reviews.c.id == rid)).first()
        return _row(r) if r else None

    def list_reviews(self, limit: int = 20) -> list[dict]:
        cols = [reviews.c.id, reviews.c.vendor_ids, reviews.c.request_kind, reviews.c.status, reviews.c.decision,
                reviews.c.requester, reviews.c.created_at]
        with self.engine.connect() as c:
            rows = c.execute(sa.select(*cols).order_by(reviews.c.created_at.desc()).limit(limit)).all()
        return [_row(r) for r in rows]

    def record_decision(self, rid: str, decision: str, actor: str, detail: dict) -> bool:
        """Set the human decision once. Returns False if the review already has one (no overwrite)."""
        with self.engine.begin() as c:
            res = c.execute(reviews.update().where(reviews.c.id == rid, reviews.c.decision.is_(None))
                            .values(decision=decision, status="DECIDED", updated_at=_now()))
            if res.rowcount != 1:
                return False
            c.execute(audit_events.insert().values(review_id=rid, actor=actor, action="HUMAN_DECISION", decision=decision,
                                                   detail=detail, created_at=_now()))
        return True

    # ---- turns
    def add_turn(self, rid: str, role: str, content: dict) -> int:
        with self.engine.begin() as c:
            res = c.execute(conversation_turns.insert().values(review_id=rid, role=role, content=content, created_at=_now()))
        return res.inserted_primary_key[0]

    def turns(self, rid: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(conversation_turns).where(conversation_turns.c.review_id == rid)
                             .order_by(conversation_turns.c.id)).all()
        return [_row(r) for r in rows]

    # ---- audit (append-only)
    def audit(self, rid: str | None, actor: str, action: str, detail: dict | None = None, decision: str | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(audit_events.insert().values(review_id=rid, actor=actor, action=action, decision=decision,
                                                   detail=detail, created_at=_now()))

    def audit_events(self, rid: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(audit_events).where(audit_events.c.review_id == rid).order_by(audit_events.c.id)).all()
        return [_row(r) for r in rows]

    def ping(self) -> bool:
        with self.engine.connect() as c:
            c.execute(sa.text("SELECT 1"))
        return True
