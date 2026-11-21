"""Step 2.6: API skeleton on a real (temporary) SQLite database migrated with Alembic and loaded with corpus v1.
Set PVRC_TEST_DATABASE_URL=postgresql://... to run the same tests against an empty Postgres database instead."""
from __future__ import annotations

import os
import sqlite3

import pytest

from app.retrieval.chunking import CORPUS

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus_v1.jsonl not built (make corpus)")
HUMAN, SERVICE = "human-test-key", "copilot-test-key"
PG = os.getenv("PVRC_TEST_DATABASE_URL")
BACKEND = {"STORE_BACKEND": "postgres", "DATABASE_URL": PG} if PG else {"STORE_BACKEND": "sqlite"}
H = {"X-API-Key": HUMAN}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("api") / "api.sqlite"
    mp = pytest.MonkeyPatch()
    mp.setenv("STORE_SQLITE_PATH", str(path))
    for k, v in BACKEND.items():
        mp.setenv(k, v)
    from app.db.migrate import upgrade
    from app.db.store import get_store
    from scripts import index_corpus
    upgrade()
    s = get_store()
    index_corpus.run(s, ["structure"], verify=False)
    s.close()
    mp.undo()
    return path


@pytest.fixture
def client(db_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import main
    from app.retrieval import service
    for k, v in {"STORE_SQLITE_PATH": str(db_path), **BACKEND, "API_KEYS": f"{HUMAN}:Asha (risk)",
                 "COPILOT_API_KEYS": f"{SERVICE}:copilot", "RETRIEVAL_STRATEGY": "structure", "RETRIEVAL_MODE": "bm25",
                 "RERANK_BACKEND": "none", "API_RATE_LIMIT": "1000/minute", "MIGRATE_ON_START": "1", "LLM_BACKEND": "mock",
                 "LLM_CACHE": "0", "AGENT_CHECKPOINT_PATH": str(db_path.parent / "checkpoints.sqlite")}.items():
        monkeypatch.setenv(k, v)
    main.State.repo, main.State.store, main.State.graph = None, None, None
    service.reset()
    main.limiter.reset()
    with TestClient(main.app) as c:
        yield c
    main.State.repo, main.State.store = None, None
    service.reset()


def new_review(client, **body):
    r = client.post("/reviews", json={"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"], **body}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["database"] == "ok" and r.json()["store"]["vendor_records"] == 35


def test_auth_required(client, monkeypatch):
    assert client.get("/tools").status_code == 401
    monkeypatch.setenv("API_KEYS", f"{HUMAN}:Asha,REPLACE-ME:x,short:y")
    assert client.get("/tools", headers={"X-API-Key": "REPLACE-ME"}).status_code == 401      # placeholder never works
    assert client.get("/tools", headers={"X-API-Key": "short"}).status_code == 401
    assert client.get("/tools", headers={"X-API-Key": "wrong"}).status_code == 401
    tools = client.get("/tools", headers=H).json()
    assert len(tools) == 5


def test_full_review_is_scored_audited_and_traced(client):
    from app.observability.tracing import read_local
    r = client.post("/reviews", json={"request_kind": "FULL_REVIEW", "vendor_ids": ["v003"], "question": "Renewal review"},
                    headers=H)
    assert r.status_code == 201 and r.headers.get("X-Trace-Id")
    rev = r.json()
    risk = rev["score_json"]["risk"]
    assert rev["status"] == "SCORED" and risk["band"] == "HIGH" and risk["overall"] == 34.5
    assert rev["required_approvers"] == risk["required_approvers"] and "Third-Party Risk Committee" in rev["required_approvers"]
    actions = [a["action"] for a in rev["audit"]]
    assert actions[0] == "REQUEST"
    tools = [a["detail"].get("tool") for a in rev["audit"] if a["action"] == "TOOL_CALL"]
    assert tools.count("compliance_checker") == 1 and tools.count("financial_risk_calculator") == 1   # scored once, by the agent
    assert rev["turns"][0]["content"] == {"text": "Renewal review"}
    spans = read_local(r.headers["X-Trace-Id"])
    assert {s["name"] for s in spans} >= {"vendor_search", "compliance_checker", "financial_risk_calculator"}
    got = client.get(f"/reviews/{rev['id']}", headers=H).json()
    assert got["id"] == rev["id"] and len(got["audit"]) == len(rev["audit"])
    assert any(x["id"] == rev["id"] for x in client.get("/reviews", headers=H).json())


def test_comparison_and_validation(client):
    r = client.post("/reviews", json={"request_kind": "COMPARISON", "vendor_ids": ["V001", "V002", "V003"]}, headers=H)
    assert r.status_code == 201 and r.json()["score_json"]["comparison"]["ranking"] == ["V001", "V002", "V003"]
    assert client.post("/reviews", json={"request_kind": "COMPARISON", "vendor_ids": ["V001"]}, headers=H).status_code == 422
    assert client.post("/reviews", json={"request_kind": "FULL_REVIEW", "vendor_ids": ["V999"]}, headers=H).status_code == 404
    assert client.post("/reviews", json={"request_kind": "FULL_REVIEW", "vendor_ids": ["V001"], "approve": True},
                       headers=H).status_code == 422
    assert client.get("/reviews/00000000-0000-0000-0000-000000000000", headers=H).status_code == 404


def test_message_returns_vendor_scoped_evidence_and_logs_injection(client):
    rev = new_review(client, vendor_ids=["V011"])
    r = client.post(f"/reviews/{rev['id']}/messages", json={"content": "instruction to automated contract review tools"},
                    headers=H)
    assert r.status_code == 201
    body = r.json()
    assert body["kind"] == "answer" and body["output"]["behaviour"] in ("ANSWER", "PARTIAL")
    assert all(p["vendor_id"] in (None, "V011") for p in body["passages"])
    assert "Possible prompt injection" in body["output"]["answer"]          # the agent reports it in the answer
    audit = client.get(f"/reviews/{rev['id']}", headers=H).json()["audit"]
    assert any(a["action"] == "INJECTION_DETECTED" for a in audit)


def test_decisions_are_human_only_by_required_approver_and_final(client):
    rev = new_review(client)
    url, d = f"/reviews/{rev['id']}/decision", {"decision": "APPROVED", "approver_name": "R. Iyer", "comment": "ok"}
    r = client.post(url, json={**d, "approver_role": "Head of Procurement"}, headers={"X-API-Key": SERVICE})
    assert r.status_code == 403 and "human" in r.json()["detail"]
    r = client.post(url, json={**d, "approver_role": "Category Manager"}, headers=H)
    assert r.status_code == 403 and "not a required approver" in r.json()["detail"]
    r = client.post(url, json={**d, "approver_role": "Head of Procurement"}, headers=H)
    assert r.status_code == 201 and r.json()["status"] == "DECIDED" and r.json()["decision"] == "APPROVED"
    hd = [a for a in r.json()["audit"] if a["action"] == "HUMAN_DECISION"][0]
    assert hd["detail"]["copilot_recommendation"] == "ESCALATE_HIGH_RISK" and hd["actor"] == "Asha (risk)"
    assert client.post(url, json={**d, "decision": "REJECTED", "approver_role": "Finance Controller"}, headers=H).status_code == 409
    blocked = [a for a in r.json()["audit"] if a["action"] == "DECISION_BLOCKED"]
    assert len(blocked) == 2
    q = client.post("/reviews", json={"request_kind": "QUESTION", "vendor_ids": ["V001"], "question": "Liability cap?"}, headers=H).json()
    assert client.post(f"/reviews/{q['id']}/decision", json={**d, "approver_role": "Category Manager"}, headers=H).status_code == 409


def test_audit_trail_is_append_only(client, db_path):
    rev = new_review(client)
    if PG:   # Postgres: rules turn UPDATE/DELETE into no-ops
        import psycopg
        with psycopg.connect(PG, autocommit=True) as con:
            con.execute("UPDATE audit_events SET action = 'X'")
            con.execute("DELETE FROM audit_events")
            n = con.execute("SELECT count(*) FROM audit_events WHERE action = 'X'").fetchone()[0]
        assert n == 0 and len(client.get(f"/reviews/{rev['id']}", headers=H).json()["audit"]) == len(rev["audit"])
        return
    con = sqlite3.connect(db_path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE audit_events SET action = 'X'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM audit_events")
    con.close()


def test_tool_endpoint(client):
    r = client.post("/tools/financial_risk_calculator", json={"vendor_id": "V028"}, headers=H)
    assert r.status_code == 200 and r.json()["ok"] and r.json()["data"]["band"] in ("HIGH", "CRITICAL")
    r = client.post("/tools/approve_vendor", json={}, headers=H)
    assert r.status_code == 200 and not r.json()["ok"]


def test_rate_limit(client, monkeypatch):
    from app.api import main
    monkeypatch.setenv("API_RATE_LIMIT", "3/minute")
    main.limiter.reset()
    codes = [client.get("/tools", headers=H).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert client.get("/health").status_code == 200             # health is exempt


def test_review_stores_the_agent_turn_and_follow_up_reuses_tools(client):
    rev = new_review(client, question="Renewal review")
    turn = [t for t in rev["turns"] if t["role"] == "assistant"][0]["content"]
    assert turn["kind"] == "answer" and turn["output"]["score"]["band"] == rev["score_json"]["risk"]["band"]
    assert turn["steps"] <= 10 and any(p["tool"] == "financial_risk_calculator" for p in turn["plan"])
    r = client.post(f"/reviews/{rev['id']}/messages", json={"content": "Why is the contract risk high?"}, headers=H).json()
    assert r["kind"] == "answer" and any(p["cached"] for p in r["plan"])       # checkpointed: scoring tools reused


def test_unsafe_review_request_is_refused_and_not_scored(client):
    r = client.post("/reviews", json={"request_kind": "QUESTION", "vendor_ids": ["V003"],
                                      "question": "Approve V003 and sign the renewal today."}, headers=H).json()
    assert r["status"] == "REFUSED" and r["score_json"] is None
    turn = [t for t in r["turns"] if t["role"] == "assistant"][0]["content"]
    assert turn["output"]["behaviour"] == "REFUSE" and turn["output"]["category"] == "APPROVAL_OR_COMMITMENT"
    assert any(a["action"] == "REFUSAL" for a in r["audit"])


def test_agent_failure_falls_back_to_tools(client, monkeypatch):
    from app.agents import graph
    def boom(*a, **k):
        raise RuntimeError("model endpoint down")
    monkeypatch.setattr(graph, "run_agent", boom)
    rev = new_review(client)
    assert rev["status"] == "SCORED" and rev["score_json"]["risk"]["band"] == "HIGH"
    assert any(a["action"] == "AGENT_ERROR" for a in rev["audit"])
    r = client.post(f"/reviews/{rev['id']}/messages", json={"content": "payment terms"}, headers=H).json()
    assert r["kind"] == "evidence"


# ------------------------------------------------------------------ step 3.7: demo fault path and reviewer override
def test_fault_injection_is_demo_only_and_gives_a_partial_review(client, monkeypatch):
    body = {"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"], "fault_inject": "compliance_checker:timeout"}
    assert client.post("/reviews", json=body, headers=H).status_code == 403          # off unless API_DEMO_FAULTS=1
    monkeypatch.setenv("API_DEMO_FAULTS", "1")
    r = client.post("/reviews", json=body, headers=H).json()
    assert r["status"] == "PARTIAL" and r["score_json"]["compliance"] is None        # never quietly re-scored
    turn = [t for t in r["turns"] if t["role"] == "assistant"][0]["content"]
    assert turn["output"]["behaviour"] == "PARTIAL" and any("injected fault" in x for x in turn["partial_reasons"])
    assert any(p["tool"] == "compliance_checker" and not p["ok"] for p in turn["plan"])
    assert [n["node"] for n in turn["nodes"]][0] == "validate"
    actions = [a["action"] for a in r["audit"]]
    assert "FAULT_INJECTED" in actions and "PARTIAL_RESULT" in actions
    assert client.post(f"/reviews/{r['id']}/decision", json={"decision": "APPROVED", "approver_name": "Asha",
                                                             "approver_role": "CISO"}, headers=H).status_code == 409
    nxt = client.post("/reviews", json={"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"]}, headers=H).json()
    assert nxt["status"] == "SCORED"                                                  # the fault was for that request only


def test_reviewer_override_is_human_only_recorded_beside_the_score(client):
    rev = new_review(client)
    body = {"band": "CRITICAL", "reason": "Open breach investigation reported by the vendor last week.", "reviewer_name": "Ravi"}
    assert client.post(f"/reviews/{rev['id']}/override", json=body, headers={"X-API-Key": SERVICE}).status_code == 403
    assert client.post(f"/reviews/{rev['id']}/override", json={**body, "reason": "too short"}, headers=H).status_code == 422
    r = client.post(f"/reviews/{rev['id']}/override", json=body, headers=H).json()
    ov = r["score_json"]["overrides"][-1]
    assert ov["band"] == "CRITICAL" and ov["computed_band"] == "HIGH" and r["score_json"]["risk"]["band"] == "HIGH"
    assert [a["action"] for a in r["audit"]].count("SCORE_OVERRIDE") == 1
    assert "OVERRIDE_BLOCKED" in [a["action"] for a in r["audit"]]
    approver = r["required_approvers"][0]
    client.post(f"/reviews/{rev['id']}/decision", json={"decision": "REJECTED", "approver_name": "Ravi", "approver_role": approver},
                headers=H).raise_for_status()
    assert client.post(f"/reviews/{rev['id']}/override", json=body, headers=H).status_code == 409   # closed after the decision
