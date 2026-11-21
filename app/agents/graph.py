"""Week 3 (3.1, 3.5): the LangGraph agent and its entry point.

    from app.agents.graph import run_agent
    state = run_agent({"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"], "question": "Renewal review"})
    state["output"]      # Report / ComparisonResult / Answer / Clarification / Refusal (app/schemas.py)

validate -> plan -> execute -> write -> critic -> finalize, with validate -> finalize for refusals and clarifications
and critic -> plan once ("replan") when a code check fails. Every node is a span in the request's trace.
AGENT_ARCH=multi selects the multi-agent graph instead (build_multi_graph, D44); the default is this single agent.

Persistence (3.5): pass review_id and a checkpointer (get_checkpointer()) to keep the state per review. A follow-up turn
on the same review starts from the stored tool results (the cache), so "why is the contract risk high?" does not
re-run the scoring tools.
"""
from __future__ import annotations

import os
import sqlite3
import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents import nodes
from app.agents.state import ReviewState, new_state
from app.observability.tracing import current_trace_id, span, start_trace
from app.tools.base import ToolContext


def _traced(name, fn):
    def run(state):
        with span(f"node.{name}", kind="span", input={"route": state.get("route"), "steps": state.get("steps")}) as s:
            upd = fn(state)
            s.end(output={k: upd[k] for k in ("route", "critic") if k in upd} or None,
                  steps=upd.get("steps", state.get("steps")), tokens=upd.get("tokens", state.get("tokens")))
            return upd
    return run


def _after_validate(state):
    return "plan" if state.get("route") == "PLAN" else "finalize"


def _after_critic(state):
    c = state.get("critic") or {}
    return "plan" if (not c.get("ok") and state.get("replanned") and not state.get("_replan_done")) else "finalize"


def _mark_replan(state):
    return {"_replan_done": True} if state.get("replanned") else {}


def _after_critic_multi(state):
    c = state.get("critic") or {}
    return "write" if (not c.get("ok") and state.get("replanned") and not state.get("_replan_done")) else "finalize"


def build_multi_graph(checkpointer=None):
    """AGENT_ARCH=multi (D44): supervisor, three specialists, writer, verifier. See app/agents/multi_agent.py."""
    from app.agents import multi_agent as M
    g = StateGraph(ReviewState)
    g.add_node("validate", _traced("validate", nodes.validate))
    g.add_node("supervisor", _traced("supervisor", M.supervisor))
    g.add_node("evidence", _traced("agent.evidence", M.evidence_agent))
    g.add_node("policy", _traced("agent.policy", M.policy_agent))
    g.add_node("risk", _traced("agent.risk", M.risk_agent))
    g.add_node("write", _traced("write", lambda s: {**nodes.write(s), **_mark_replan(s)}))
    g.add_node("verifier", _traced("verifier", M.verifier))
    g.add_node("critic", _traced("critic", M.critic))
    g.add_node("finalize", _traced("finalize", nodes.finalize))
    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", lambda s: "supervisor" if s.get("route") == "PLAN" else "finalize",
                            {"supervisor": "supervisor", "finalize": "finalize"})
    for a, b in (("supervisor", "evidence"), ("evidence", "policy"), ("policy", "risk"), ("risk", "write"),
                 ("write", "verifier"), ("verifier", "critic")):
        g.add_edge(a, b)
    g.add_conditional_edges("critic", _after_critic_multi, {"write": "write", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def build_graph(checkpointer=None, arch: str | None = None):
    """The agent graph for AGENT_ARCH (or `arch`): the single agent by default, the multi-agent graph for "multi"."""
    from app.agents.state import arch as current_arch
    if (arch or current_arch()) == "multi":
        return build_multi_graph(checkpointer)
    g = StateGraph(ReviewState)
    g.add_node("validate", _traced("validate", nodes.validate))
    g.add_node("plan", _traced("plan", lambda s: {**nodes.plan(s), **_mark_replan(s)}))
    g.add_node("execute", _traced("execute", nodes.execute))
    g.add_node("write", _traced("write", nodes.write))
    g.add_node("critic", _traced("critic", nodes.critic))
    g.add_node("finalize", _traced("finalize", nodes.finalize))
    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", _after_validate, {"plan": "plan", "finalize": "finalize"})
    g.add_edge("plan", "execute")
    g.add_edge("execute", "write")
    g.add_edge("write", "critic")
    g.add_conditional_edges("critic", _after_critic, {"plan": "plan", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


ReviewState.__annotations__["_replan_done"] = bool      # internal flag: the one replan has happened


@lru_cache(maxsize=2)
def _graph_for(arch: str):
    return build_graph(arch=arch)


def _default_graph():
    from app.agents.state import arch
    return _graph_for(arch())


def get_checkpointer():
    """SQLite file next to the store by default; Postgres when STORE_BACKEND=postgres and the package is installed."""
    if os.getenv("STORE_BACKEND", "sqlite").lower() in ("postgres", "pg"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
            from app.db.models import database_url
            url = database_url().replace("postgresql+psycopg://", "postgresql://")
            conn = Connection.connect(url, autocommit=True, prepare_threshold=0)
            saver = PostgresSaver(conn)
            saver.setup()
            return saver
        except Exception as e:                      # package missing or database unreachable: fall back, never fail a review
            import logging
            logging.getLogger(__name__).warning("Postgres checkpointer unavailable (%s); using the SQLite file", e)
    from langgraph.checkpoint.sqlite import SqliteSaver
    from app.retrieval.chunking import ROOT
    path = os.getenv("AGENT_CHECKPOINT_PATH", str(ROOT / "data" / "processed" / "checkpoints.sqlite"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def run_agent(request: dict, review_id: str = "", ctx: ToolContext | None = None, graph=None, user: str | None = None) -> dict:
    """Run one request (or one follow-up turn of a review). Returns the final state; the answer is state['output']."""
    token = nodes.CTX.set(ctx or ToolContext())
    t0 = time.time()
    try:
        with start_trace("agent.run", input=request, metadata={"review_id": review_id}, user_id=user):
            if graph is not None and review_id and getattr(graph, "checkpointer", None):
                config = {"configurable": {"thread_id": review_id}}
                prev = graph.get_state(config).values or {}
                history = list(prev.get("history") or [])
                if prev.get("request"):
                    history.append({"question": prev["request"].get("question"), "behaviour": (prev.get("output") or {}).get("behaviour")})
                req = dict(request)
                if not req.get("vendor_ids") and prev.get("request"):
                    req["vendor_ids"] = prev["request"].get("vendor_ids") or []
                state = new_state(req, review_id, history=history, tool_cache=prev.get("tool_cache"))
                state["_replan_done"] = False
                out = graph.invoke(state, config)
            else:
                state = new_state(request, review_id)
                state["_replan_done"] = False
                out = (graph or _default_graph()).invoke(state)
            out["latency_ms"] = int((time.time() - t0) * 1000)
            out["trace_id"] = current_trace_id()
            return out
    finally:
        nodes.CTX.reset(token)
