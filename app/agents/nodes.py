"""the five agent nodes plus finalize.

  validate  rules first (refusals, vendor resolution, clarify cases), LLM router only for unclear QUESTION requests
  plan      a template with a deterministic default plan; the LLM only fills slots (search queries) for questions
  execute   runs the plan through run_tool(): step cap, loop prevention, cross-turn cache, fault injection
  write     reports and comparisons are assembled from tool output; the LLM writes only the explanation / answer text
  critic    code checks: citations retrieved, numbers grounded, no PII, injection reported, schema valid -> replan once
  finalize  refusal / clarification / final answer, PII masking, audit events
Nodes take and return plain dicts; the tool context (store, audit sink) comes from a context variable set by run_agent.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
from datetime import date

from app.agents.glossary import EXTERNAL_DOC_TYPES, expand_for, named_sources, order_by_named_source
from app.agents.key_facts import key_facts
from app.agents import prompts, rules
from app.agents.llm import BudgetExceeded, call_json, check_budget
from app.agents.state import (MAX_STEPS, MULTI_MAX_STEPS, AgentPlan, PlannerOutput, RouteDecision, ToolCall, WriterAnswer, WriterSummary, log)
from app.llm_client import ContextTooLong
from app.safety import NON_EVIDENCE_TYPES, redact_pii
from app.schemas import Answer, Clarification, ComparisonResult, Refusal, Report
from app.tools.base import DEFAULT_AS_OF, ToolContext, run_tool

CTX: contextvars.ContextVar = contextvars.ContextVar("agent_ctx", default=None)
FAULT: contextvars.ContextVar = contextvars.ContextVar("agent_fault", default="")   # per-request fault (API demo mode)
WEAK = ("MISSING", "EXPIRED", "CONTRADICTORY", "EXPIRING_SOON")
ITEM_QUERY = {"SOC2_TYPE2": "SOC 2 Type II report audit period", "ISO27001": "ISO 27001 certificate validity",
              "SECURITY_ATTESTATION": "SOC 2 ISO 27001 attestation", "CAIQ": "CAIQ questionnaire answers",
              "PENTEST": "penetration test findings", "CYBER_INSURANCE": "cyber insurance certificate cover",
              "BCP_DR_TEST": "disaster recovery test report", "DPA": "data processing agreement",
              "FINANCIALS": "audited financial statements"}


def ctx() -> ToolContext:
    c = CTX.get()
    if c is None:
        c = ToolContext()
        CTX.set(c)
    return c


def audit(action: str, detail: dict):
    c = ctx()
    if c.audit:
        try:
            c.audit(action, detail)
        except Exception:
            pass


# ------------------------------------------------------------------ tool calls with guards
def _key(tool: str, args: dict) -> str:
    return hashlib.sha1((tool + json.dumps(args, sort_keys=True, default=str)).encode()).hexdigest()[:16]


def call_tool(state: dict, upd: dict, tool: str, args: dict) -> dict:
    """Run one tool through run_tool() with the agent's guards. Returns the result dict (also appended to upd)."""
    cache = upd.setdefault("tool_cache", dict(state.get("tool_cache") or {}))
    results = upd.setdefault("tool_results", list(state.get("tool_results") or []))
    k = _key(tool, args)
    if any(r.get("key") == k for r in results):              # same call twice in one request: never re-run (loop guard)
        prev = next(r for r in results if r.get("key") == k)
        st = upd.setdefault("json_stats", dict(state.get("json_stats") or {}))
        st["repeated_calls_prevented"] = st.get("repeated_calls_prevented", 0) + 1
        return prev
    if k in cache:                                            # earlier turn of this review: reuse, costs no step
        r = {**cache[k], "cached": True}
        results.append(r)
        return r
    check_budget(state, upd, 0)
    upd["steps"] = upd.get("steps", state.get("steps", 0)) + 1
    fault = FAULT.get() or os.getenv("FAULT_INJECT", "")     # e.g. document_search:timeout (tool-failure demo path)
    if fault and fault.split(":")[0] == tool:
        r = {"tool": tool, "arguments": args, "ok": False, "data": None, "latency_ms": 0, "key": k,
             "error": f"injected fault: {fault.split(':', 1)[-1]}"}
    else:
        res = run_tool(tool, args, ctx())
        cap = MULTI_MAX_STEPS if state.get("arch") == "multi" else MAX_STEPS
        if not res.ok and (res.error or "").startswith("timed out") and upd.get("steps", state.get("steps", 0)) < cap:
            # every tool is read-only, so one retry after a timeout is safe; it costs a step and is counted
            upd["steps"] = upd.get("steps", state.get("steps", 0)) + 1
            st = upd.setdefault("json_stats", dict(state.get("json_stats") or {}))
            st["tool_retries"] = st.get("tool_retries", 0) + 1
            res = run_tool(tool, args, ctx())
        r = {"tool": tool, "arguments": args, "ok": res.ok, "data": res.data, "error": res.error,
             "latency_ms": res.latency_ms, "key": k}
        if res.ok:
            cache[k] = {kk: vv for kk, vv in r.items() if kk != "cached"}
    results.append(r)
    return r


def result(state: dict, tool: str) -> dict | None:
    """Latest successful data of a tool in this request."""
    for r in reversed(state.get("tool_results") or []):
        if r["tool"] == tool and r["ok"]:
            return r["data"]
    return None


# ------------------------------------------------------------------ 1 validate
def validate(state: dict) -> dict:
    req = state["request"]
    q = req.get("question") or ""
    kind = req.get("request_kind", "QUESTION")
    upd: dict = {}
    cat = rules.refusal(q)
    reason = "rule"
    mixed = kind == "FULL_REVIEW" and rules.mixed_approval(q)
    if cat is None and kind == "QUESTION" and rules.risky(q) and os.getenv("AGENT_LLM_ROUTER", "1") == "1":
        try:
            d, u = call_json({**state, **upd}, prompts.router_messages(q), RouteDecision, "router", max_tokens=60)
            if d and d.refuse and d.category == "APPROVAL_OR_COMMITMENT" and rules.status_question(q):
                upd["errors"] = list(state.get("errors") or []) + ["router APPROVAL refusal overridden: status question"]
                d = None                                                 # asking WHETHER something is approved is answerable
            upd.update(u)
            if d and d.refuse and d.category:
                cat, reason = d.category, "llm router"
        except (BudgetExceeded, ContextTooLong) as e:
            upd["errors"] = list(state.get("errors") or []) + [f"router skipped: {e}"]
    if cat:
        return {**upd, "route": "REFUSE", "refusal_category": cat, "route_reason": reason,
                "trace": log(state, "validate", route="REFUSE", category=cat, by=reason)}

    vids = [v.strip().upper() for v in req.get("vendor_ids") or [] if v.strip()]
    vendors: list[dict] = []
    if vids:
        args = {"vendor_ids": vids, "include_record": True} if len(vids) == 1 else {"vendor_ids": vids}
        r = call_tool(state, upd, "vendor_search", args)                # single vendor: the plan reuses this record
        if not r["ok"]:                                                 # a failed lookup is not an unknown vendor: never ask
            reason = f"vendor lookup failed: {r.get('error') or 'unknown error'}"
            return {**upd, "route": "FAIL", "partial_reasons": list(state.get("partial_reasons") or []) + [reason],
                    "trace": log({**state, **upd}, "validate", route="FAIL", error=r.get("error"))}
        vendors = (r["data"] or {}).get("matches", [])
        unknown = [v for v in vids if v not in {m["vendor_id"] for m in vendors}]
        if unknown:
            return _clarify(state, upd, f"I can't find vendor {', '.join(unknown)}. Which vendor do you mean?", [])
    elif kind in ("FULL_REVIEW", "COMPARISON"):
        names = rules.candidate_names(q)
        for n in names:
            r = call_tool(state, upd, "vendor_search", {"query": n, "limit": 5})
            d = r["data"] or {}
            if d.get("resolution") == "unique":
                vendors.append(d["matches"][0])
            elif d.get("resolution") == "ambiguous":
                return _clarify(state, upd, f"Several vendors match '{n}'. Which one do you mean?",
                                [f"{m['vendor_id']} {m['legal_name']}" for m in d["matches"]][:5])
        if not vendors:
            hint = rules.category_hint(q)
            if hint:
                r = call_tool(state, upd, "vendor_search", {"category": hint, "limit": 5})
                d = r["data"] or {}
                if d.get("resolution") == "unique":
                    vendors = d["matches"][:1]
                elif d.get("matches"):
                    return _clarify(state, upd, "Which vendor do you mean? Several match that description.",
                                    [f"{m['vendor_id']} {m['legal_name']}" for m in d["matches"]][:5])
        if not vendors:
            named = f" named '{names[0]}'" if names else ""
            return _clarify(state, upd, f"I can't find a registered vendor{named}. Which vendor do you mean? "
                                        "It may need to be onboarded first.", [])
        vids = [v["vendor_id"] for v in vendors]
    if kind == "COMPARISON":
        if len(vendors) == 1:
            g = vendors[0]["comparison_group"]
            r = call_tool(state, upd, "vendor_search", {"comparison_group": g, "limit": 10})
            peers = [m for m in (r["data"] or {}).get("matches", []) if m["vendor_id"] != vendors[0]["vendor_id"]]
            if not peers:
                return _clarify(state, upd, f"{vendors[0]['legal_name']} has no registered peer in its comparison group "
                                            f"({g}). Which vendor should it be compared with?", [])
            return _clarify(state, upd, f"Which vendors should {vendors[0]['legal_name']} be compared with?",
                            [f"{m['vendor_id']} {m['legal_name']}" for m in peers][:5])
        groups = {v["comparison_group"] for v in vendors}
        if len(groups) > 1:
            return _clarify(state, upd, "These vendors are in different comparison groups (" + ", ".join(sorted(groups)) +
                            ") and do not offer the same service, so a risk ranking would not be like-for-like. "
                            "Do you want separate reviews, or a comparison within one group?",
                            [f"{v['vendor_id']} {v['legal_name']} ({v['comparison_group']})" for v in vendors])
    note = None
    if mixed:
        note = "Approval declined: the copilot cannot approve vendors; the decision is routed to the required approvers."
        audit("MIXED_REQUEST_APPROVAL_DECLINED", {"question": q})
    req2 = {**req, "vendor_ids": vids}
    return {**upd, "route": "PLAN", "request": req2, "vendors": vendors, "mixed_request_note": note,
            "trace": log({**state, **upd}, "validate", route="PLAN", vendors=vids)}


def _clarify(state, upd, question, options):
    return {**upd, "route": "CLARIFY", "clarify": {"question": question[:300], "options": options[:5]},
            "trace": log({**state, **upd}, "validate", route="CLARIFY")}


# ------------------------------------------------------------------ 2 plan
def default_plan(state: dict, template: str, queries: list[str] | None = None) -> AgentPlan:
    req = state["request"]
    vids = req.get("vendor_ids") or []
    as_of = str(req.get("as_of") or DEFAULT_AS_OF)
    q = (req.get("question") or "")[:500]
    calls: list[ToolCall] = []
    if template == "FULL_REVIEW":
        v = vids[0]
        calls = [ToolCall(tool="vendor_search", arguments={"vendor_ids": [v], "include_record": True}, purpose="vendor record"),
                 ToolCall(tool="compliance_checker", arguments={"vendor_id": v, "as_of": as_of}, purpose="tier and checklist"),
                 ToolCall(tool="financial_risk_calculator", arguments={"vendor_id": v, "as_of": as_of}, purpose="scorecard")]
    elif template == "COMPARISON":
        calls = [ToolCall(tool="vendor_compare", arguments={"vendor_ids": vids, "as_of": as_of}, purpose="normalised scorecard")]
    elif template in ("VENDOR_QUESTION", "FOLLOW_UP"):
        v = vids[0]
        calls = [ToolCall(tool="vendor_search", arguments={"vendor_ids": [v], "include_record": True}, purpose="vendor record"),
                 ToolCall(tool="compliance_checker", arguments={"vendor_id": v, "as_of": as_of}, purpose="evidence statuses"),
                 ToolCall(tool="financial_risk_calculator", arguments={"vendor_id": v, "as_of": as_of}, purpose="scorecard")]
        for s in (queries or [q])[:2]:
            calls.append(ToolCall(tool="document_search", arguments={"query": s[:500], "vendor_ids": vids[:3]}, purpose="evidence passages"))
    else:  # POLICY_QUESTION: up to 2 planner queries + the question itself (the planner's keywords can miss the source)
        named = named_sources(q)
        qs = list(dict.fromkeys((queries or [])[:2] + [q]))[:3]
        for i, s in enumerate(qs):
            args = {"query": s[:500]}
            if vids:
                args["vendor_ids"] = vids[:3]
            if named and i == len(qs) - 1 and not vids:  # the question names a regulation: search only external sources once
                args["doc_types"] = EXTERNAL_DOC_TYPES
            calls.append(ToolCall(tool="document_search", arguments=args, purpose="policy / regulation passages"))
    for c in calls:
        if c.tool == "document_search":
            c.arguments["query"] = expand_for(c.arguments["query"], q)
    return AgentPlan(template=template, calls=calls[:6], source="default")


def plan(state: dict) -> dict:
    req = state["request"]
    kind = req.get("request_kind", "QUESTION")
    upd: dict = {}
    if kind == "FULL_REVIEW":
        p = default_plan(state, "FULL_REVIEW")
    elif kind == "COMPARISON":
        p = default_plan(state, "COMPARISON")
    else:
        template = ("FOLLOW_UP" if state.get("history") else "VENDOR_QUESTION") if req.get("vendor_ids") else "POLICY_QUESTION"
        queries = None
        vendor = (state.get("vendors") or [None])[0]
        try:
            out, u = call_json(state, prompts.planner_messages(req.get("question") or "", vendor), PlannerOutput, "planner",
                               max_tokens=150)
            upd.update(u)
            if out:
                queries = [s for s in out.search_queries if len(s.strip()) >= 3][:3] or None
        except (BudgetExceeded, ContextTooLong) as e:
            upd["errors"] = list(state.get("errors") or []) + [f"planner skipped: {e}"]
        if state.get("replanned"):                # second attempt after a failed check: widen the search with the question
            queries = list(dict.fromkeys((queries or []) + [req.get("question") or ""]))[:3]
        p = default_plan(state, template, queries)
        if queries:
            p.source = "llm"
    return {**upd, "plan": p.model_dump(), "trace": log({**state, **upd}, "plan", template=p.template, source=p.source,
                                                        calls=[c.tool for c in p.calls])}


# ------------------------------------------------------------------ 3 execute
def execute(state: dict) -> dict:
    upd: dict = {}
    partial = list(state.get("partial_reasons") or [])
    p = AgentPlan.model_validate(state["plan"])
    try:
        for c in p.calls:
            r = call_tool(state, upd, c.tool, c.arguments)
            if not r["ok"]:
                partial.append(f"{c.tool} failed: {r['error']}")
        if p.template == "FULL_REVIEW" and not state.get("replanned"):     # evidence passages for the weakest items
            comp = result({**state, **upd}, "compliance_checker")
            vendor = result({**state, **upd}, "vendor_search")
            if comp and vendor and vendor.get("matches"):
                name = vendor["matches"][0]["legal_name"].split(" Pvt")[0].split(" Ltd")[0]
                weak = [r["item"] for r in comp["checklist"] if r["required"] and r["status"] in WEAK][:2] or ["CAIQ"]
                for item in weak:
                    r = call_tool(state, upd, "document_search", {"query": f"{name} {ITEM_QUERY.get(item, item)}"[:500],
                                                                  "vendor_ids": [comp["vendor_id"]], "k": 4})
                    if not r["ok"]:
                        partial.append(f"document_search failed: {r['error']}")
    except BudgetExceeded as e:
        partial.append(str(e))
    passages = fuse_passages(upd.get("tool_results", state.get("tool_results") or []))
    return {**upd, "passages": passages, "partial_reasons": partial,
            "trace": log({**state, **upd}, "execute", tools=[r["tool"] for r in upd.get("tool_results", [])
                                                                if r.get("key")][-6:], failures=len(partial))}


def fuse_passages(tool_results: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal-rank fusion over all document_search results of the request. Concatenating them would let the first
    search fill all 6 prompt slots; fusion gives each search's top hits a place."""
    score: dict[str, float] = {}
    first: dict[str, dict] = {}
    for r in tool_results:
        if r["tool"] == "document_search" and r["ok"]:
            for rank, psg in enumerate(r["data"]["passages"], 1):
                pid = psg["passage_id"]
                first.setdefault(pid, psg)
                score[pid] = score.get(pid, 0.0) + 1.0 / (k + rank)
    order = sorted(first, key=lambda pid: -score[pid])        # stable: ties keep first-seen order
    return [first[pid] for pid in order]


# ------------------------------------------------------------------ 4 write
def _sources(state: dict) -> dict[str, dict]:
    """doc_id -> {title, date, type} for every document the agent may cite (passages + record + tool sources)."""
    out: dict[str, dict] = {}
    for p in state.get("passages") or []:
        out[p["doc_id"]] = {"title": p["title"], "document_date": p.get("document_date"), "type": p["evidence_type"]}
    vs = result(state, "vendor_search")
    for docs in (vs or {}).get("documents", {}).values():
        for d in docs:
            out.setdefault(d["doc_id"], {"title": d["title"], "document_date": d.get("document_date"), "type": d["doc_type"]})
    for tool in ("compliance_checker", "financial_risk_calculator"):
        d = result(state, tool) or {}
        for f in d.get("findings", []):
            for s in f["sources"]:
                out.setdefault(s["doc_id"], {"title": s.get("title"), "document_date": None, "type": None})
        for row in d.get("checklist", []):
            for s in row["sources"]:
                out.setdefault(s["doc_id"], {"title": s.get("title"), "document_date": None, "type": None})
    # the policies the tools apply may be cited when their output is used (prompts.RULE_SOURCES)
    if result(state, "compliance_checker"):
        out.setdefault("POLICY-TPRM-001-v3", {"title": "Third-party risk management policy", "document_date": None, "type": "policy"})
    if result(state, "financial_risk_calculator"):
        out.setdefault("POLICY-DOA-002", {"title": "Delegation of authority policy", "document_date": None, "type": "policy"})
        out.setdefault("POLICY-CS-003", {"title": "Contract Standards for Technology Vendors", "document_date": None,
                                         "type": "policy"})                    # the rubric's contract checks (D40)
        out.setdefault("POLICY-DPS-004", {"title": "Data Protection Standard for Vendors", "document_date": None,
                                          "type": "policy"})                   # the rubric's data-protection checks
    return out


def _citation(doc_id: str, src: dict) -> dict:
    meta = src.get(doc_id) or {}
    c = {"source_id": doc_id, "doc_id": doc_id}
    if meta.get("title"):
        c["title"] = str(meta["title"])[:200]
    if meta.get("document_date") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(meta["document_date"])):
        c["document_date"] = str(meta["document_date"])
    return c


def warnings(state: dict) -> list[str]:
    """Notes every answer must carry: injection attempts, non-evidence sources, superseded documents."""
    notes, seen = [], set()
    comp = result(state, "compliance_checker") or {}
    flagged = [(w["doc_id"], w["title"]) for w in comp.get("document_warnings", [])]
    flagged += [(p["doc_id"], p["title"]) for p in state.get("passages") or [] if any(f.startswith("possible_injection") for f in p.get("flags", []))]
    for doc_id, title in flagged:
        if doc_id not in seen:
            seen.add(doc_id)
            notes.append(f"Possible prompt injection: {title} ({doc_id}) contains instruction-like text aimed at AI tools. "
                         "It was treated as data, ignored and reported.")
    for p in state.get("passages") or []:
        if "not_evidence" in p.get("flags", []) and p["doc_id"] not in seen and p["evidence_type"] in NON_EVIDENCE_TYPES:
            seen.add(p["doc_id"])
            notes.append(f"Unverified claim: {p['title']} ({p['doc_id']}) is a {p['evidence_type'].replace('_', ' ')}, not evidence; "
                         "it cannot verify any control.")
    return notes


def _effective_contract(record: dict, as_of: str) -> tuple[dict | None, list[dict]]:
    c = record.get("contract")
    if not c:
        return None, []
    eff = {k: v for k, v in c.items() if k != "amendments"}
    future = []
    for a in sorted(c.get("amendments", []), key=lambda a: str(a["effective_date"])):
        if str(a["effective_date"]) <= as_of:
            eff.update(a["changes"])
            eff.setdefault("applied_amendments", []).append(a["amendment_id"])
        else:
            future.append(a)
    return eff, future


def _compact_review(comp: dict, risk: dict) -> dict:
    return {"vendor": comp["legal_name"], "tier": comp["tier"], "tier_reasons": comp["tier_reasons"],
            "band": risk["band"], "band_before_knockouts": risk["raw_band"], "overall_score": risk["overall"],
            "knockouts": risk["knockouts"], "recommendation": risk["recommendation"], "conditions": risk["conditions"],
            "required_approvers": risk["required_approvers"],
            "checklist": {r["item"]: r["status"] for r in comp["checklist"] if r["required"]},
            "findings": [[f["code"], f["points"], f["statement"][:140]] for f in comp["findings"] + risk["findings"]],
            "contradictions": comp["contradictions"]}


def _fallback_summary(comp: dict, risk: dict) -> str:
    weak = [f"{r['item']} {r['status'].lower().replace('_', ' ')}" for r in comp["checklist"] if r["required"] and r["status"] in WEAK]
    top = sorted(comp["findings"] + risk["findings"], key=lambda f: -f["points"])[:3]
    s = (f"{comp['legal_name']} is tier {comp['tier']} and rated {risk['band']} (score {risk['overall']}). "
         f"Recommendation: {risk['recommendation'].replace('_', ' ').lower()}. ")
    if risk["knockouts"]:
        s += f"Knockouts: {', '.join(risk['knockouts'])}. "
    if weak:
        s += f"Evidence gaps: {', '.join(weak)}. "
    if top:
        s += "Main findings: " + "; ".join(f["statement"] for f in top) + "."
    return s[:800]


def write(state: dict) -> dict:
    req = state["request"]
    kind = req.get("request_kind", "QUESTION")
    upd: dict = {}
    partial = list(state.get("partial_reasons") or [])
    notes = warnings(state)
    src = _sources(state)
    feedback = (state.get("critic") or {}).get("problems") if state.get("replanned") else None
    if kind == "FULL_REVIEW":
        comp, risk = result(state, "compliance_checker"), result(state, "financial_risk_calculator")
        if not (comp and risk):
            draft = _partial_answer(state, src, partial or ["the scoring tools did not return a result"], notes)
            return {**upd, "draft": draft, "trace": log(state, "write", kind="partial")}
        compact = _compact_review(comp, risk)
        summary = None
        msgs = prompts.summary_messages(compact, state.get("passages") or [], 4)
        if feedback:
            msgs[-1]["content"] += "\nYour previous summary had these problems; fix them: " + "; ".join(feedback)[:400]
        try:
            out, u = call_json(state, msgs, WriterSummary, "write.summary", max_tokens=400)
            upd.update(u)
            summary = out.summary if out else None
        except ContextTooLong:
            try:
                out, u = call_json(state, prompts.summary_messages(compact, [], 0), WriterSummary, "write.summary", max_tokens=400)
                upd.update(u)
                summary = out.summary if out else None
            except (BudgetExceeded, ContextTooLong) as e:
                partial.append(str(e))
        except BudgetExceeded as e:
            partial.append(str(e))
        findings = [{"code": f["code"], "dimension": f["dimension"], "severity": f["severity"], "kind": "interpretation",
                     "statement": f["statement"][:500], "needs_legal_review": f["needs_legal_review"],
                     "citations": [_citation(s["doc_id"], {s["doc_id"]: {"title": s.get("title")}, **src}) for s in f["sources"]]}
                    for f in comp["findings"] + risk["findings"]]
        gaps = [f"{r['item']}: {r['status']} - {r['reason']}"[:300] for r in comp["checklist"] if r["required"] and r["status"] in WEAK]
        gaps += [f"CAIQ contradiction: {c}" for c in comp["contradictions"]]
        gaps += notes
        if state.get("mixed_request_note"):
            gaps.append(state["mixed_request_note"].replace("the required approvers", ", ".join(risk["required_approvers"])))
        gaps += [f"Incomplete: {p}" for p in partial]
        vendor = (result(state, "vendor_search") or {}).get("matches", [{}])[0]
        draft = {"behaviour": "PARTIAL" if partial else "ANSWER", "vendor_id": comp["vendor_id"],
                 "review_type": req.get("review_type") or vendor.get("review_type") or "NEW_VENDOR", "as_of": str(comp["as_of"]),
                 "tier": comp["tier"], "summary": (summary or _fallback_summary(comp, risk))[:800], "findings": findings,
                 "checklist": [{"item": r["item"], "required": r["required"], "status": r["status"], "evidence_ids": r["evidence_ids"],
                                "reason": r["reason"][:300]} for r in comp["checklist"]],
                 "score": {"overall": risk["overall"], "band": risk["band"], "raw_band": risk["raw_band"],
                           "dimensions": [{"dimension": d["dimension"], "points": d["points"], "weight": d["weight"],
                                           "applicable": d["applicable"], "triggered_rules": d["triggered_rules"]} for d in risk["dimensions"]],
                           "knockouts": risk["knockouts"], "rubric_version": risk["rubric_version"]},
                 "recommendation": risk["recommendation"], "conditions": risk["conditions"],
                 "approval": {"required_approvers": risk["required_approvers"]}, "gaps_and_uncertainties": gaps,
                 "partial_reason": "; ".join(partial)[:500] if partial else None}
        return {**upd, "draft": draft, "partial_reasons": partial,
                "trace": log({**state, **upd}, "write", kind="report", summary_by="llm" if summary else "template")}
    if kind == "COMPARISON":
        cmp_ = result(state, "vendor_compare")
        if not cmp_:
            return {**upd, "draft": _partial_answer(state, src, partial or ["vendor_compare failed"], notes),
                    "trace": log(state, "write", kind="partial")}
        draft = {"behaviour": "PARTIAL" if partial else "ANSWER",
                 "rows": [{"vendor_id": r["vendor_id"], "tier": r["tier"], "overall": r["overall"], "band": r["band"],
                           "dimension_points": r["dimension_points"], "key_findings": r["key_findings"]} for r in cmp_["rows"]],
                 "caveats": cmp_["caveats"] + [f"Ranking: {' < '.join(cmp_['ranking'])} (lower risk first). {cmp_['ranking_rule']}"]}
        return {**upd, "draft": draft, "trace": log(state, "write", kind="comparison")}
    # QUESTION
    record = (result(state, "vendor_search") or {}).get("records", {})
    record = next(iter(record.values()), None)
    as_of = str(req.get("as_of") or DEFAULT_AS_OF)
    eff, future = _effective_contract(record, as_of) if record else (None, [])
    comp_, risk_ = result(state, "compliance_checker"), result(state, "financial_risk_calculator")
    facts = prompts.vendor_facts(record, comp_, risk_, eff, future)
    question = req.get("question") or ""
    key = key_facts(question, comp_, risk_) if os.getenv("AGENT_KEY_FACTS", "1") == "1" else []
    passages, named = order_by_named_source(state.get("passages") or [], question)
    if not req.get("vendor_ids"):                 # a rule question: drafts, e-mails, web pages are not rules (D40)
        passages = [p for p in passages if p.get("evidence_type") not in NON_EVIDENCE_TYPES] or passages
    out = None
    for n in (6, 4, 2, 0):                       # shrink the context until the prompt fits the window
        citable = _citable(state, passages[:n], key)
        msgs = prompts.question_messages(question, as_of, facts, passages, n, key=key, named=named, citable=citable,
                                         handoffs=state.get("handoffs") if state.get("arch") == "multi" else None)
        if feedback:
            msgs[-1]["content"] += "\nYour previous answer had these problems; fix them: " + "; ".join(feedback)[:400]
        try:
            out, u = call_json(state, msgs, WriterAnswer, "write.answer", max_tokens=500, schema=_answer_schema(citable))
            upd.update(u)
            break
        except ContextTooLong:
            continue
        except BudgetExceeded as e:
            partial.append(str(e))
            break
    if out is None:
        return {**upd, "draft": _partial_answer(state, src, partial or ["the answer could not be generated"], notes),
                "partial_reasons": partial, "trace": log({**state, **upd}, "write", kind="partial")}
    answer = scrub_labels(out.answer.strip())
    if notes:
        answer = (answer + " " + " ".join(notes))[:1200]
    cites = [c for c in dict.fromkeys(out.citations)]
    cits = [_citation(c, src) for c in cites][:6]
    if not cits:
        cits = [_citation(passages[0]["doc_id"], src)] if passages else [{"source_id": "tool:none"}]
    draft = {"behaviour": "PARTIAL" if partial else "ANSWER", "answer": answer, "citations": cits}
    return {**upd, "draft": draft, "partial_reasons": partial, "trace": log({**state, **upd}, "write", kind="answer")}


_LABEL = r"(?:KEY[_ ]FACTS|FACTS|PASSAGES|CITABLE DOC IDS)"
_SCRUB = [
    (re.compile(r"\s*\[\s*doc_ids?\s*:[^\]]*\]", re.I), ""),                                  # "[doc_ids: A, B]"
    (re.compile(r",?\s*(?:with|and)\s+doc_ids?\s*:\s*[A-Z0-9_,\s-]+(?=[.;]|$)", re.I), ""),     # "with doc_ids: A, B."
    (re.compile(rf",?\s*\(?(?:as\s+)?(?:stated|indicated|shown|listed|given|noted)\s+in\s+(?:the\s+)?{_LABEL}\)?"), ""),
    (re.compile(rf"\b(?:according to|per|from)\s+(?:the\s+)?{_LABEL}\b"), "according to the review's checks"),
    (re.compile(rf"\b([Tt])he\s+{_LABEL}\b"), lambda m: f"{m.group(1)}he review's checks"),
    (re.compile(r"\bKEY[_ ]FACTS\b"), "the review's checks"),
    (re.compile(r"\b(as of|on)\s+AS_OF\b:?\s*", re.I), r"\1 "),                       # "as of AS_OF 2026-11-01"
    (re.compile(r"\bAS_OF\b:?\s*"), ""),
]


def scrub_labels(text: str) -> str:
    """D43: the 8B writer sometimes names the prompt's internal sections ("as stated in the KEY_FACTS with doc_ids: ...",
    "as of AS_OF 2026-11-01"). Removed in code, not by a prompt rule: a rule sentence changes every question's prompt,
    and the first benchmark run showed that rewording alone moves unrelated answers. Citations stay in the citations list."""
    for pat, rep in _SCRUB:
        text = pat.sub(rep, text)
    text = re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()
    return text[:1].upper() + text[1:] if text else text


def _citable(state: dict, shown: list[dict], key: list[dict]) -> list[str]:
    """Doc ids the answer may cite: the passages in the prompt, the key facts' sources, then the rest of what the agent
    retrieved (record documents, tool and rule sources). Capped so the schema stays small."""
    ids = ([p["doc_id"] for p in shown] + [d for k in key for d in k["doc_ids"]]
           + [d for h in (state.get("handoffs") or []) for f in h.get("findings", []) for d in f.get("doc_ids", [])]
           + list(_sources(state)))
    allowed = set(_sources(state)) | {p["doc_id"] for p in shown}
    return [d for d in dict.fromkeys(ids) if d in allowed][:40]


def _answer_schema(citable: list[str]) -> dict | None:
    """WriterAnswer's JSON schema with citations limited to real ids (D40). Under constrained decoding the endpoint
    cannot emit an invented id; without it, the critic still checks. AGENT_CITATION_ENUM=0 turns this off (ablation)."""
    if not citable or os.getenv("AGENT_CITATION_ENUM", "1") != "1":
        return None
    schema = WriterAnswer.model_json_schema()
    schema["properties"]["citations"]["items"] = {"type": "string", "enum": citable}
    return schema


def _partial_answer(state: dict, src: dict, reasons: list[str], notes: list[str]) -> dict:
    """Safe default when tools or the model fail: say what is known and why the answer is partial (never guess)."""
    top = (state.get("passages") or [])[:3]
    reasons = list(dict.fromkeys(reasons))                   # a replan can report the same failure twice
    text = ("I could not complete this answer (" + "; ".join(reasons)[:300] + "). "
            + ("The most relevant sources found are: " + "; ".join(f"{p['title']} ({p['doc_id']})" for p in top) + ". " if top else "")
            + " ".join(notes))
    cites = [_citation(p["doc_id"], src) for p in top] or [{"source_id": "tool:none"}]
    return {"behaviour": "PARTIAL", "answer": text[:1200], "citations": cites}


# ------------------------------------------------------------------ 5 critic (code checks)
NUM = re.compile(r"(?<![\w.\-/])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)(?![\w\-/])")   # skips ids like DOA-002, 800-161


def _nums(text: str) -> set[str]:
    out = set()
    for m in NUM.findall(text or ""):
        v = m.replace(",", "")
        out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def _schema_for(kind: str, draft: dict):
    """A PARTIAL draft is always an Answer (what we have, what is missing), whatever the request kind."""
    if draft.get("behaviour") == "PARTIAL" and "answer" in draft:
        return Answer
    return {"FULL_REVIEW": Report, "COMPARISON": ComparisonResult}.get(kind, Answer)


def critic_enabled() -> bool:
    """AGENT_CRITIC=0 (ablation): no code checks, no replan, no last-resort fixes in finalize. Safety (PII masking) and
    schema validation stay on: they are not the critic's."""
    return os.getenv("AGENT_CRITIC", "1") == "1"


def critic(state: dict) -> dict:
    if not critic_enabled():
        return {"critic": {"ok": True, "problems": [], "skipped": True}, "trace": log(state, "critic", ok=True, skipped=True)}
    d = state.get("draft") or {}
    kind = state["request"].get("request_kind", "QUESTION")
    problems: list[str] = []
    src = _sources(state)
    if "answer" in d and d.get("behaviour") != "PARTIAL":
        allowed = set(src)
        bad = [c["doc_id"] for c in d["citations"] if c.get("doc_id") and c["doc_id"] not in allowed]
        if bad:
            problems.append(f"cited documents that were not retrieved: {', '.join(bad)}")
        grounding = json.dumps([result(state, t) for t in ("vendor_search", "compliance_checker", "financial_risk_calculator")],
                               default=str) + " ".join(p["text"] for p in state.get("passages") or []) + (state["request"].get("question") or "")
        ungrounded = sorted(_nums(d["answer"]) - _nums(grounding) - {"1", "2", "3", "0"})
        if ungrounded:
            problems.append(f"numbers not found in the facts or passages: {', '.join(ungrounded[:5])}")
    if kind == "FULL_REVIEW" and d.get("summary"):
        risk = result(state, "financial_risk_calculator") or {}
        named = {b for b in ("LOW", "MEDIUM", "HIGH", "CRITICAL") if re.search(rf"\b{b}\b", d["summary"])}
        if named - {risk.get("band"), risk.get("raw_band")}:
            problems.append(f"summary names band(s) {', '.join(sorted(named))} but the scorecard band is {risk.get('band')}")
        if re.search(r"\b(approved|i approve|we approve)\b", d["summary"], re.I):
            problems.append("summary must not state or imply an approval")
    blob = json.dumps(d, ensure_ascii=False)
    if redact_pii(blob)[1]:
        problems.append("personal identifiers in the output")
    try:
        _schema_for(kind, d).model_validate(d)
    except Exception as e:
        problems.append(f"schema: {str(e)[:200]}")
    ok = not problems
    upd = {"critic": {"ok": ok, "problems": problems}, "trace": log(state, "critic", ok=ok, problems=problems)}
    if not ok and not state.get("replanned"):
        upd["replanned"] = True
        audit("REPLAN", {"problems": problems})
    return upd


# ------------------------------------------------------------------ finalize
def finalize(state: dict) -> dict:
    route = state.get("route")
    if route == "REFUSE":
        cat = state["refusal_category"]
        out = Refusal(behaviour="REFUSE", category=cat, message=rules.MESSAGES[cat], route_to=rules.ROUTE_TO[cat]).model_dump(mode="json")
        audit("REFUSAL", {"category": cat, "by": state.get("route_reason"), "question": state["request"].get("question")})
    elif route == "CLARIFY":
        c = state["clarify"]
        out = Clarification(behaviour="CLARIFY", question=c["question"], options=c["options"]).model_dump(mode="json")
        audit("CLARIFICATION", c)
    elif route == "FAIL":
        out = Answer.model_validate(_partial_answer(state, {}, state.get("partial_reasons") or ["a tool failed"], [])).model_dump(mode="json")
        audit("PARTIAL_RESULT", {"reasons": state.get("partial_reasons")})
    else:
        d = dict(state.get("draft") or {})
        problems = (state.get("critic") or {}).get("problems") or []
        if "answer" in d:                         # last-resort fixes after the one replan: drop unretrieved citations, mask PII
            if critic_enabled():
                allowed = set(_sources(state))
                good = [c for c in d["citations"] if not c.get("doc_id") or c["doc_id"] in allowed]
                d["citations"] = good or [{"source_id": "tool:none"}]
            d["answer"] = redact_pii(d["answer"])[0]
        kind = state["request"].get("request_kind", "QUESTION")
        schema = _schema_for(kind, d)
        try:
            out = schema.model_validate(d).model_dump(mode="json")
        except Exception as e:
            out = _partial_answer(state, _sources(state), [f"output failed validation: {str(e)[:120]}"], warnings(state))
            out = Answer.model_validate(out).model_dump(mode="json")
        if problems and state.get("replanned"):
            audit("CRITIC_UNRESOLVED", {"problems": problems})
        if out.get("behaviour") == "PARTIAL":
            audit("PARTIAL_RESULT", {"reasons": state.get("partial_reasons")})
    return {"output": out, "trace": log(state, "finalize", behaviour=out.get("behaviour"))}
