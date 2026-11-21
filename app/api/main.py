"""FastAPI skeleton (step 2.6). Run: uvicorn app.api.main:app --reload   (docs at http://localhost:8000/docs)

Routes
  GET  /health                       liveness + database + store + retrieval mode (no key needed)
  POST /reviews                      open a review case; runs the deterministic tools (scorecard) for full reviews/comparisons
  GET  /reviews, GET /reviews/{id}   case, conversation turns and audit trail
  POST /reviews/{id}/messages        follow-up question; Week 2 answers with retrieved evidence (the agent arrives in Week 3)
  POST /reviews/{id}/decision        HUMAN decision by one of the required approvers; never by the copilot's service key
  GET  /tools, POST /tools/{name}    tool catalogue and direct tool calls (for the UI and debugging)

Security: every route except /health needs X-API-Key. API_KEYS holds human users' keys, COPILOT_API_KEYS the agent's
service keys (format key or key:name, comma-separated). Service keys cannot record decisions. Requests are rate limited
per key (API_RATE_LIMIT, default 60/minute). Every request opens a trace; the trace id is returned in X-Trace-Id.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

from app.api.repository import Repository
from app.observability.tracing import current_trace_id, start_trace
from app.tools.base import DEFAULT_AS_OF, ToolContext, describe_tools, run_tool

load_dotenv()


# ------------------------------------------------------------------ auth
class Principal(BaseModel):
    name: str
    kind: Literal["human", "service"]
    key_id: str


def _parse_keys(env: str, kind: str) -> dict[str, Principal]:
    out = {}
    for i, item in enumerate(x.strip() for x in os.getenv(env, "").split(",") if x.strip()):
        key, _, name = item.partition(":")
        if "REPLACE" in key or len(key) < 8:          # placeholder or trivially guessable: never accepted
            continue
        out[key] = Principal(name=name or f"{kind}-{i + 1}", kind=kind, key_id=hashlib.sha256(key.encode()).hexdigest()[:8])
    return out


def principal(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Principal:
    keys = {**_parse_keys("API_KEYS", "human"), **_parse_keys("COPILOT_API_KEYS", "service")}
    if not keys:
        raise HTTPException(503, "API_KEYS is not configured on the server")
    for k, p in keys.items():
        if x_api_key and secrets.compare_digest(x_api_key, k):
            return p
    raise HTTPException(401, "missing or invalid X-API-Key")


def _rate_key(request: Request) -> str:
    k = request.headers.get("X-API-Key")
    return "key:" + hashlib.sha256(k.encode()).hexdigest()[:12] if k else "ip:" + (request.client.host if request.client else "?")


limiter = Limiter(key_func=_rate_key, default_limits=[lambda: os.getenv("API_RATE_LIMIT", "60/minute")])


# ------------------------------------------------------------------ app state
class State:
    repo: Repository | None = None
    store = None
    graph = None


def agent_enabled() -> bool:
    """API_AGENT=1 (default): reviews and messages run the Week 3 agent. 0: the Week 2 tool-only skeleton."""
    return os.getenv("API_AGENT", "1") == "1"


def agent_graph():
    if State.graph is None:
        from app.agents.graph import build_graph, get_checkpointer
        State.graph = build_graph(get_checkpointer())
    return State.graph


def run_review_agent(rid: str, request: dict, who: "Principal", ctx: ToolContext, fault: str = "") -> dict | None:
    """Run one agent turn for a review (checkpointed per review id). Errors never fail the request: they are audited
    and the caller falls back to the deterministic tools."""
    from app.agents.graph import run_agent
    from app.agents.nodes import FAULT
    token = FAULT.set(fault or "")
    try:
        return run_agent(request, review_id=rid, ctx=ctx, graph=agent_graph(), user=who.name)
    except Exception as e:                                 # noqa: BLE001  (fallback path is tested)
        get_repo().audit(rid, "copilot", "AGENT_ERROR", {"error": f"{type(e).__name__}: {str(e)[:300]}"})
        return None
    finally:
        FAULT.reset(token)


def demo_faults_enabled() -> bool:
    """API_DEMO_FAULTS=1 lets a request ask for one failing tool (the tool-failure demo path). Off by default."""
    return os.getenv("API_DEMO_FAULTS", "0") == "1"


def agent_turn(st: dict) -> dict:
    from app.agents.state import max_steps
    """What an assistant turn stores: the output, the passages shown as evidence, the plan and the cost."""
    return {"kind": "answer", "output": st["output"], "behaviour": st["output"].get("behaviour"),
            "passages": [{k: p.get(k) for k in ("doc_id", "title", "vendor_id", "evidence_type", "document_date", "status",
                                                "flags", "text", "score")} for p in (st.get("passages") or [])[:6]],
            "plan": [{"tool": r["tool"], "arguments": r["arguments"], "ok": r["ok"], "cached": bool(r.get("cached")),
                      "error": r.get("error")} for r in st.get("tool_results") or []],
            "steps": st.get("steps"), "llm_calls": st.get("llm_calls"), "tokens": st.get("tokens"),
            "replanned": bool(st.get("replanned")), "latency_ms": st.get("latency_ms"), "trace_id": st.get("trace_id"),
            "template": (st.get("plan") or {}).get("template"), "route": st.get("route"),
            "partial_reasons": list(dict.fromkeys(st.get("partial_reasons") or [])), "critic": st.get("critic"),
            "nodes": [{k: t.get(k) for k in ("node", "step", "route", "template", "ok", "problems", "error") if t.get(k) is not None}
                      for t in st.get("trace") or []],
            # multi-agent graph (D44): who ran, what each specialist handed over, the verifier's verdict
            "arch": st.get("arch", "single"), "step_cap": max_steps(st),
            "agents": (st.get("supervisor") or {}).get("tasks") if st.get("arch") == "multi" else None,
            "handoffs": st.get("handoffs") if st.get("arch") == "multi" else None,
            "verifier": st.get("verifier") if st.get("arch") == "multi" else None}


def get_repo() -> Repository:
    if State.repo is None:
        State.repo = Repository()
    return State.repo


def get_store():
    if State.store is None:
        from app.db.store import get_store as gs
        State.store = gs()
    return State.store


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("MIGRATE_ON_START", "1") == "1":
        from app.db.migrate import upgrade
        upgrade()
    yield


app = FastAPI(title="Procurement & Vendor Risk Copilot API", version="0.2.0 (Week 2 skeleton)", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
def _rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail": f"rate limit exceeded: {exc.detail}"}, status_code=429)


@app.middleware("http")
async def _trace_header(request: Request, call_next):
    response = await call_next(request)
    tid = getattr(request.state, "trace_id", None)
    if tid:
        response.headers["X-Trace-Id"] = tid
    return response


def ctx_for(review_id: str | None, who: Principal) -> ToolContext:
    """Tool context whose audit sink writes TOOL_CALL / INJECTION_DETECTED events for this review (actor = copilot)."""
    repo = get_repo()
    return ToolContext(store=get_store(), audit=lambda action, detail: repo.audit(review_id, "copilot", action,
                                                                                 {**detail, "on_behalf_of": who.name}))


def tool_or_502(name: str, args: dict, ctx: ToolContext) -> dict:
    r = run_tool(name, args, ctx)
    if not r.ok:
        raise HTTPException(502 if "unknown vendor" not in (r.error or "") else 404, f"{name} failed: {r.error}")
    return r.data


# ------------------------------------------------------------------ request / response models
class ReviewCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_kind: Literal["FULL_REVIEW", "COMPARISON", "QUESTION"]
    vendor_ids: list[str] = Field(min_length=1, max_length=3)
    review_type: Optional[Literal["NEW_VENDOR", "RENEWAL"]] = None
    question: Optional[str] = Field(default=None, max_length=2000)
    requester_role: Literal["procurement_analyst", "risk_reviewer", "category_manager", "legal_finance_reviewer"] = "procurement_analyst"
    as_of: Optional[date] = None
    fault_inject: Optional[str] = Field(default=None, pattern=r"^(vendor_search|document_search|compliance_checker|"
                                        r"financial_risk_calculator|vendor_compare):[a-z_]{1,30}$",
                                        description="demo only (API_DEMO_FAULTS=1): make one tool fail for this request")

    @field_validator("vendor_ids", mode="before")
    @classmethod
    def _up(cls, v):
        return [x.strip().upper() for x in v] if isinstance(v, list) else v


class OverrideCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    reason: str = Field(min_length=20, max_length=2000, description="why the reviewer disagrees with the computed band")
    reviewer_name: str = Field(min_length=2, max_length=100)


class MessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=2000)


class DecisionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["APPROVED", "APPROVED_WITH_CONDITIONS", "REJECTED", "ESCALATED", "MORE_INFORMATION_REQUESTED"]
    approver_name: str = Field(min_length=2, max_length=100)
    approver_role: str = Field(min_length=2, max_length=100, description="must be one of the review's required approvers")
    comment: str = Field(default="", max_length=2000)


# ------------------------------------------------------------------ routes
@app.get("/health")
@limiter.exempt
def health(request: Request):
    out = {"status": "ok", "version": app.version}
    try:
        get_repo().ping()
        out["database"] = "ok"
    except Exception as e:
        out["status"], out["database"] = "degraded", f"{type(e).__name__}: {str(e)[:120]}"
    try:
        c = get_store().counts()
        out["store"] = {k: c[k] for k in ("documents", "vendor_records", "chunks")}
    except Exception as e:
        out["status"], out["store"] = "degraded", f"{type(e).__name__}: {str(e)[:120]}"
    from app.retrieval import service
    out["retrieval"] = {"strategy": os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx"), "mode_setting": os.getenv("RETRIEVAL_MODE", "auto"),
                        "note": service.mode_note() or None}
    return out


@app.get("/tools")
def list_tools(request: Request, who: Principal = Depends(principal)):
    return describe_tools()


@app.post("/tools/{name}")
def call_tool(name: str, args: dict, request: Request, who: Principal = Depends(principal)):
    with start_trace(f"tool.{name}", input=args, user_id=who.name) as t:
        request.state.trace_id = getattr(t, "id", None)
        r = run_tool(name, args, ctx_for(None, who))
    return r.model_dump(mode="json")


@app.post("/reviews", status_code=201)
def create_review(body: ReviewCreate, request: Request, who: Principal = Depends(principal)):
    repo = get_repo()
    with start_trace("review.create", input=body.model_dump(mode="json"), user_id=who.name) as t:
        request.state.trace_id = getattr(t, "id", None)
        if body.request_kind == "COMPARISON" and len(body.vendor_ids) < 2:
            raise HTTPException(422, "a comparison needs 2-3 vendor_ids")
        if len(set(body.vendor_ids)) != len(body.vendor_ids):
            raise HTTPException(422, "vendor_ids must be different vendors")
        if body.fault_inject and not demo_faults_enabled():
            raise HTTPException(403, "fault injection is a demo feature: start the API with API_DEMO_FAULTS=1")
        ctx = ToolContext(store=get_store())
        found = tool_or_502("vendor_search", {"vendor_ids": body.vendor_ids}, ctx)
        known = {m["vendor_id"]: m for m in found["matches"]}
        unknown = [v for v in body.vendor_ids if v not in known]
        if unknown:
            raise HTTPException(404, f"unknown vendor id(s): {', '.join(unknown)}; use POST /tools/vendor_search to find them")
        review_type = body.review_type or known[body.vendor_ids[0]]["review_type"]
        as_of = body.as_of or DEFAULT_AS_OF
        rid = repo.create_review(vendor_ids=body.vendor_ids, review_type=review_type, request_kind=body.request_kind, as_of=as_of,
                                 question=body.question, requester=who.name, requester_role=body.requester_role,
                                 trace_id=current_trace_id())
        repo.audit(rid, who.name, "REQUEST", {"request": body.model_dump(mode="json"), "key_id": who.key_id})
        fault = body.fault_inject or ""
        if fault:
            repo.audit(rid, who.name, "FAULT_INJECTED", {"fault": fault, "note": "demo: this tool is made to fail for this request"})
        if body.question:
            repo.add_turn(rid, "user", {"text": body.question})
        ctx = ctx_for(rid, who)
        st = None
        if agent_enabled():
            req = {**body.model_dump(mode="json", exclude={"fault_inject"}), "review_type": review_type, "as_of": str(as_of)}
            st = run_review_agent(rid, req, who, ctx, fault)
            if st:
                repo.add_turn(rid, "assistant", agent_turn(st))
        from app.agents.nodes import result
        faulty = fault.split(":")[0] if fault else ""

        def got(tool: str, args: dict):
            """The agent's result for a tool, else one direct run (recovery). A demo-injected fault is never retried:
            the review then stays PARTIAL instead of quietly scoring."""
            if st and result(st, tool):
                return result(st, tool)
            return None if tool == faulty else tool_or_502(tool, args, ctx)

        score, status = None, "OPEN"
        if st and st.get("route") == "REFUSE":
            status = "REFUSED"                           # nothing is scored for a refused request
        elif body.request_kind == "FULL_REVIEW":
            v = body.vendor_ids[0]
            score = {"compliance": got("compliance_checker", {"vendor_id": v, "as_of": str(as_of)}),
                     "risk": got("financial_risk_calculator", {"vendor_id": v, "as_of": str(as_of)})}
            status = "SCORED" if all(score.values()) else "PARTIAL"
        elif body.request_kind == "COMPARISON":
            score = {"comparison": got("vendor_compare", {"vendor_ids": body.vendor_ids, "as_of": str(as_of)})}
            status = "SCORED" if score["comparison"] else "PARTIAL"
        repo.update_review(rid, score_json=score, status=status)
    return review_view(rid)


def review_view(rid: str) -> dict:
    repo = get_repo()
    r = repo.get_review(rid)
    if r is None:
        raise HTTPException(404, "review not found")
    r["required_approvers"] = ((r.get("score_json") or {}).get("risk") or {}).get("required_approvers", [])
    vs = run_tool("vendor_search", {"vendor_ids": r.get("vendor_ids") or [r["vendor_id"]]}, ToolContext(store=get_store()))
    r["vendors"] = [{k: m.get(k) for k in ("vendor_id", "legal_name", "category", "review_type")}
                    for m in ((vs.data or {}).get("matches", []) if vs.ok else [])]
    r["vendor"] = r["vendors"][0] if r["vendors"] else None
    r["turns"] = repo.turns(rid)
    r["audit"] = repo.audit_events(rid)
    return r


@app.get("/reviews")
def list_reviews(request: Request, limit: int = 20, who: Principal = Depends(principal)):
    return get_repo().list_reviews(min(max(limit, 1), 100))


@app.get("/reviews/{rid}")
def get_review(rid: str, request: Request, who: Principal = Depends(principal)):
    return review_view(rid)


@app.post("/reviews/{rid}/messages", status_code=201)
def post_message(rid: str, body: MessageCreate, request: Request, who: Principal = Depends(principal)):
    repo = get_repo()
    r = repo.get_review(rid)
    if r is None:
        raise HTTPException(404, "review not found")
    with start_trace("review.message", input=body.content, user_id=who.name, metadata={"review_id": rid}) as t:
        request.state.trace_id = getattr(t, "id", None)
        repo.add_turn(rid, "user", {"text": body.content})
        if agent_enabled():
            req = {"request_kind": "QUESTION", "vendor_ids": r["vendor_ids"] or [r["vendor_id"]], "question": body.content,
                   "review_type": r.get("review_type"), "as_of": str(r.get("as_of") or DEFAULT_AS_OF),
                   "requester_role": r.get("requester_role") or "procurement_analyst"}
            st = run_review_agent(rid, req, who, ctx_for(rid, who))
            if st:
                content = agent_turn(st)
                repo.add_turn(rid, "assistant", content)
                return content
        res = run_tool("document_search", {"query": body.content, "vendor_ids": r["vendor_ids"] or [r["vendor_id"]]}, ctx_for(rid, who))
        if res.ok:
            content = {"kind": "evidence", "passages": res.data["passages"], "mode": res.data["mode"], "note":
                       "The most relevant cited passages for your question (the agent is off or failed: API_AGENT=0 "
                       "or see the AGENT_ERROR audit event)." + (" " + res.data["note"] if res.data.get("note") else "")}
        else:
            content = {"kind": "error", "note": f"retrieval failed: {res.error}"}
        repo.add_turn(rid, "assistant", content)
    return content


@app.post("/reviews/{rid}/override", status_code=201)
def post_override(rid: str, body: OverrideCreate, request: Request, who: Principal = Depends(principal)):
    """A reviewer disagrees with the computed band and says why. Recorded next to the score, never instead of it: the
    tools' numbers stay as computed, and approvers see both. Human keys only; every override is an audit event."""
    repo = get_repo()
    if who.kind != "human":
        repo.audit(rid, who.name, "OVERRIDE_BLOCKED", {"reason": "service key", "attempted": body.band})
        raise HTTPException(403, "only a human reviewer can override the band, never the copilot")
    r = repo.get_review(rid)
    if r is None:
        raise HTTPException(404, "review not found")
    risk = (r.get("score_json") or {}).get("risk")
    if r["request_kind"] != "FULL_REVIEW" or not risk:
        raise HTTPException(409, "only a scored full review has a band to override")
    if r.get("decision"):
        raise HTTPException(409, "a decision is already recorded; the band can no longer be overridden")
    ov = {"band": body.band, "computed_band": risk["band"], "reason": body.reason, "reviewer_name": body.reviewer_name,
          "by": who.name, "key_id": who.key_id, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    score = dict(r["score_json"])
    score["overrides"] = list(score.get("overrides") or []) + [ov]
    repo.update_review(rid, score_json=score)
    repo.audit(rid, who.name, "SCORE_OVERRIDE", ov)
    return review_view(rid)


@app.post("/reviews/{rid}/decision", status_code=201)
def post_decision(rid: str, body: DecisionCreate, request: Request, who: Principal = Depends(principal)):
    repo = get_repo()
    if who.kind != "human":
        repo.audit(rid, who.name, "DECISION_BLOCKED", {"reason": "service key", "attempted": body.decision})
        raise HTTPException(403, "decisions can only be recorded by a human approver, never by the copilot")
    r = repo.get_review(rid)
    if r is None:
        raise HTTPException(404, "review not found")
    if r["request_kind"] != "FULL_REVIEW" or not r.get("score_json") or r.get("status") != "SCORED":
        raise HTTPException(409, "only a fully scored full review can receive an approval decision"
                                 + (" (this one is PARTIAL: re-run the review)" if r.get("status") == "PARTIAL" else ""))
    required = r["score_json"]["risk"]["required_approvers"]
    if body.approver_role not in required:
        repo.audit(rid, who.name, "DECISION_BLOCKED", {"reason": "not a required approver", "approver_role": body.approver_role})
        raise HTTPException(403, f"{body.approver_role!r} is not a required approver for this review; required: {', '.join(required)}")
    with start_trace("review.decision", input=body.model_dump(), user_id=who.name, metadata={"review_id": rid}) as t:
        request.state.trace_id = getattr(t, "id", None)
        ok = repo.record_decision(rid, body.decision, who.name, {**body.model_dump(), "key_id": who.key_id,
                                                                 "copilot_recommendation": r["score_json"]["risk"]["recommendation"]})
    if not ok:
        raise HTTPException(409, "this review already has a decision; decisions are never overwritten")
    return review_view(rid)
