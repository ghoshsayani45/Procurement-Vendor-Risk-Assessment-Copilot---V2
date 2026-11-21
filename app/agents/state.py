"""the agent's state and plan schemas. Frozen interfaces: the UI, API, safety tests and eval build on these.

ReviewState is a plain dict (LangGraph TypedDict state). Every node reads it and returns the keys it changes.
The final answer is `state["output"]`: one of the Week 1 output schemas (Report, ComparisonResult, Answer,
Clarification, Refusal in app/schemas.py), so the Week 1 grader and failure classifier work unchanged.
"""
from __future__ import annotations

import os
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field

TemplateId = Literal["FULL_REVIEW", "COMPARISON", "VENDOR_QUESTION", "POLICY_QUESTION", "FOLLOW_UP"]
Route = Literal["PLAN", "REFUSE", "CLARIFY", "FAIL"]   # FAIL: a tool needed to start failed -> PARTIAL


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["vendor_search", "document_search", "compliance_checker", "financial_risk_calculator", "vendor_compare"]
    arguments: dict
    purpose: str = Field(default="", max_length=200)


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template: TemplateId
    calls: list[ToolCall] = Field(min_length=1, max_length=6)
    source: Literal["llm", "default"] = "default"     # default = the template's deterministic plan
    note: str = ""


class PlannerOutput(BaseModel):
    """What the LLM planner may fill (and nothing else): a template id and its slots."""
    model_config = ConfigDict(extra="forbid")
    template: Literal["VENDOR_QUESTION", "POLICY_QUESTION"]
    search_queries: list[str] = Field(min_length=1, max_length=3)
    need_scorecard: bool = False


class WriterAnswer(BaseModel):
    """What the LLM writer returns for questions: the answer text and the ids of the sources it used."""
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=1200)
    citations: list[str] = Field(min_length=1, max_length=6)


class WriterSummary(BaseModel):
    """What the LLM writes for a full review: the explanation only (every number comes from the tools)."""
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=800)


class RouteDecision(BaseModel):
    """LLM fallback router for requests the rules could not classify."""
    model_config = ConfigDict(extra="forbid")
    refuse: bool
    category: Optional[Literal["APPROVAL_OR_COMMITMENT", "LEGAL_ADVICE", "ACCESS_CONTROL_BYPASS", "PII_EXPOSURE", "SECRETS",
                               "SCORE_MANIPULATION", "OUT_OF_SCOPE"]] = None


class ReviewState(TypedDict, total=False):
    # request
    review_id: str
    request: dict                    # request_kind, vendor_ids, question, review_type, requester_role, as_of
    history: list[dict]              # earlier turns of this review (follow-ups)
    # routing and plan
    route: Route
    route_reason: str
    vendors: list[dict]              # resolved vendors (vendor_search matches)
    refusal_category: Optional[str]
    clarify: dict                    # {"question":..., "options": [...]}
    mixed_request_note: Optional[str]
    plan: dict                       # AgentPlan
    replanned: bool
    # execution
    tool_results: list[dict]         # [{"tool", "arguments", "ok", "data", "error", "latency_ms", "cached"}]
    tool_cache: dict                 # hash(tool, args) -> result (persists across turns: follow-ups reuse it)
    passages: list[dict]             # deduped passages from document_search
    # writing and checking
    draft: dict                      # output before the critic
    critic: dict                     # {"ok": bool, "problems": [...]}
    output: dict                     # final output (Week 1 schemas)
    # guards and accounting
    steps: int                       # tool calls + LLM calls so far (cap: AGENT_MAX_STEPS = 10)
    llm_calls: int
    tokens: int                      # includes replayed tokens of cached calls (cost as if uncached; the quota is not charged)
    cached_llm_calls: int
    cached_latency_ms: int           # original latency of cached calls, added back for latency metrics
    cached_estimated: bool           # a cached call came from an old cache file: tokens estimated, latency unknown
    json_stats: dict                 # constrained / first_try / repaired / retried / defaulted counts
    partial_reasons: list[str]
    errors: list[str]
    trace: list[dict]                # node-by-node log for the UI's plan view and the failure taxonomy
    trace_id: Optional[str]


MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
TOKEN_CAPS = {"FULL_REVIEW": int(os.getenv("AGENT_TOKENS_FULL_REVIEW", "22000")),
              "COMPARISON": int(os.getenv("AGENT_TOKENS_QUESTION", "15000")),
              "QUESTION": int(os.getenv("AGENT_TOKENS_QUESTION", "15000")),
              "REFUSE": 2000}


def token_cap(state: dict) -> int:
    if state.get("route") == "REFUSE":
        return TOKEN_CAPS["REFUSE"]
    return TOKEN_CAPS.get(state["request"].get("request_kind", "QUESTION"), TOKEN_CAPS["QUESTION"])


def new_state(request: dict, review_id: str = "", history: list | None = None, tool_cache: dict | None = None) -> dict:
    return ReviewState(review_id=review_id, request=dict(request), history=history or [], tool_results=[], passages=[],
                       tool_cache=dict(tool_cache or {}), steps=0, llm_calls=0, tokens=0, partial_reasons=[], errors=[],
                       trace=[], replanned=False, json_stats={"calls": 0, "constrained": 0, "first_try": 0, "repaired": 0,
                                                              "retried": 0, "retry_succeeded": 0, "defaulted": 0})


def log(state: dict, node: str, **info: Any) -> list[dict]:
    """Append one line to the node log (returned as the new `trace` value)."""
    return list(state.get("trace", [])) + [{"node": node, "step": state.get("steps", 0), **info}]
