"""D45: the chat API on a real (temporary) SQLite database migrated with Alembic (0003) and loaded with corpus v1.
Uses the offline stand-in model (LLM_BACKEND=mock): the routes, persistence, ownership, audit and grounding fields are
what is tested, not the wording of LLM answers."""
from __future__ import annotations

import pytest

from app.retrieval.chunking import CORPUS

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus_v1.jsonl not built (make corpus)")
HUMAN, OTHER = "human-test-key", "other-human-key"
H, O = {"X-API-Key": HUMAN}, {"X-API-Key": OTHER}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("chatapi") / "api.sqlite"
    mp = pytest.MonkeyPatch()
    mp.setenv("STORE_SQLITE_PATH", str(path))
    mp.setenv("STORE_BACKEND", "sqlite")
    from app.db.migrate import current, upgrade
    from app.db.store import get_store
    from scripts import index_corpus
    upgrade()
    assert current() == "0003"
    s = get_store()
    index_corpus.run(s, ["structure"], verify=False)
    s.close()
    mp.undo()
    return path


@pytest.fixture
def client(db_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import main
    from app.chatbot import engine, portfolio, router
    from app.retrieval import service
    for k, v in {"STORE_SQLITE_PATH": str(db_path), "STORE_BACKEND": "sqlite", "API_KEYS": f"{HUMAN}:Asha,{OTHER}:Ravi",
                 "COPILOT_API_KEYS": "", "RETRIEVAL_STRATEGY": "structure", "RETRIEVAL_MODE": "bm25", "RERANK_BACKEND": "none",
                 "API_RATE_LIMIT": "1000/minute", "CHAT_RATE_LIMIT": "1000/minute", "MIGRATE_ON_START": "1",
                 "LLM_BACKEND": "mock", "LLM_CACHE": "0", "AGENT_CHECKPOINT_PATH": str(db_path.parent / "cp.sqlite")}.items():
        monkeypatch.setenv(k, v)
    main.State.repo, main.State.store, main.State.graph = None, None, None
    service.reset()
    engine.clear_cache()
    portfolio.reset()
    router.reset()
    main.limiter.reset()
    with TestClient(main.app) as c:
        yield c
    main.State.repo, main.State.store = None, None
    service.reset()


def test_chat_needs_a_key(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 401


def test_chat_session_keeps_state_and_history(client):
    r = client.post("/chat", json={"message": "Which vendor has the highest risk score?"}, headers=H)
    assert r.status_code == 201, r.text
    t = r.json()
    sid = t["session_id"]
    assert t["route"] == "PORTFOLIO" and t["grounding"]["method"] == "deterministic" and t["table"][0]["vendor_id"] == "V017"
    assert t["citations"] and t["trace_id"] and r.headers.get("X-Trace-Id")
    r = client.post("/chat", json={"message": "What is its recommendation?", "session_id": sid}, headers=H)
    t2 = r.json()
    assert t2["session_id"] == sid and t2["route"] == "VENDOR" and t2["vendor_ids"] == ["V017"]        # focus persisted in the DB
    s = client.get(f"/chat/sessions/{sid}", headers=H).json()
    assert [m["role"] for m in s["messages"]] == ["user", "assistant", "user", "assistant"]
    assert s["state"]["focus_vendor_ids"] == ["V017"]
    assert any(x["id"] == sid for x in client.get("/chat/sessions", headers=H).json())


def test_chat_sessions_are_private(client):
    sid = client.post("/chat", json={"message": "hi"}, headers=H).json()["session_id"]
    assert client.get(f"/chat/sessions/{sid}", headers=O).status_code == 403
    assert client.post("/chat", json={"message": "hi", "session_id": sid}, headers=O).status_code == 403
    assert all(x["id"] != sid for x in client.get("/chat/sessions", headers=O).json())
    assert client.get("/chat/sessions/00000000-0000-0000-0000-000000000000", headers=H).status_code == 404
    assert client.post("/chat", json={"message": "x" * 2001}, headers=H).status_code == 422


def test_chat_refusal_is_audited(client):
    r = client.post("/chat", json={"message": "Approve Payvanta's renewal now."}, headers=H).json()
    assert r["behaviour"] == "REFUSE" and r["refusal_category"] == "APPROVAL_OR_COMMITMENT" and r["tool_calls"] == 0
    from app.api.main import get_repo
    import sqlalchemy as sa
    from app.db.models import audit_events
    with get_repo().engine.connect() as c:
        rows = c.execute(sa.select(audit_events).where(audit_events.c.action == "CHAT_REFUSAL")).all()
    assert any((dict(x._mapping)["detail"] or {}).get("chat_session_id") == r["session_id"] for x in rows)


def test_chat_kb_reports_its_sources(client):
    kb = client.get("/chat/kb", headers=H).json()
    assert kb["chunks"] > 100 and "docs/spec.md" in kb["files"] and kb["mode"] in ("bm25", "hybrid")


def test_reviews_still_work_alongside_chat(client):
    r = client.post("/reviews", json={"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"]}, headers=H)
    assert r.status_code == 201 and r.json()["score_json"]["risk"]["band"] == "HIGH"
