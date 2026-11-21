"""The chat turn: route -> execute -> write -> check -> answer, for any question about the copilot system (D45).

    turn = ask("Is Kestrion's cyber insurance still valid?", session, ctx=ToolContext(store=...))

Routes (app/chatbot/router.py) and how each one stays grounded:
  HELP        fixed text.
  REFUSE      the agent's refusal rules and messages (spec section 6); audited.
  CLARIFY     unknown vendor name: ask, never guess a vendor.
  VENDOR      overview questions: a snapshot written by code from compliance_checker + financial_risk_calculator.
              Other questions: the Week 3 agent (validate -> plan -> execute -> write -> critic) runs; its answer is used
              only if its own critic passed and every number in it appears in its tool results or passages; otherwise
              the tools' key facts (app/agents/key_facts.py) are shown instead, as written by code.
  COMPARISON  vendor_compare, text written by code.
  PORTFOLIO   every vendor's tool results, filtered and aggregated by code (app/chatbot/portfolio.py).
  SYSTEM /    the project documents (app/chatbot/knowledge.py) and the internal policies and regulations in the corpus
  POLICY      (document_search) are searched; no answer is attempted when the best passage covers too little of the
              question (NOT_FOUND). The LLM writes from at most 6 passages; the code checks citations, numbers, sentence
              support and PII (app/chatbot/grounding.py), asks once more with the problems, then keeps only the supported
              sentences, and if nothing survives quotes the sources verbatim instead.
Every turn is one trace (chat.turn) with a span per tool and LLM call, and is capped at CHAT_MAX_STEPS own steps.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from app import llm_client
from app.agents.glossary import EXTERNAL_DOC_TYPES, expand_for, named_sources, order_by_named_source
from app.agents.key_facts import key_facts
from app.agents.llm import BudgetExceeded, call_json
from app.agents.state import RouteDecision
from app.chatbot import grounding, knowledge, portfolio, router
from app.chatbot.prompts import ChatAnswer, answer_schema, knowledge_messages, portfolio_messages
from app.llm_client import ContextTooLong
from app.observability.tracing import current_trace_id, span, start_trace
from app.safety import NON_EVIDENCE_TYPES, redact_pii
from app.tools.base import DEFAULT_AS_OF, ToolContext, run_tool

RULE_DOC_TYPES = ["internal_policy", "procurement_policy", "guidance", "contract_template"]
HISTORY_Q = re.compile(r"\b(v2(\.1)?|older|old version|previous version|superseded|history|changed|used to)\b", re.I)
ELLIPTICAL = re.compile(r"^\s*(and|also|what about|how about|why|and why|on the|for the)\b|^\W*\w+(\W+\w+){0,3}\W*$", re.I)
HELP_TEXT = """I answer questions about the vendor-risk copilot from its own records, and I cite where every answer comes from:

- **A vendor**: "Is Kestrion's cyber insurance still valid?", "Who has to approve Opticrest?", "Give me an overview of Voxenta."
- **The vendor portfolio**: "Which vendors are rated CRITICAL?", "Which vendors have an expired SOC 2 report?", "Show the number of vendors in each risk band."
- **Policies and regulations**: "Within how many hours must a vendor report a data breach?", "What evidence does a tier 2 vendor need?"
- **How the copilot works**: "How is the risk score calculated?", "What retrieval does the copilot use?", "Why was the multi-agent design rejected?"

I never approve, sign or change anything, and when the records don't answer a question I say so instead of guessing."""
SUGGEST = {
    "HELP": ["Which vendors are rated CRITICAL?", "How is the overall risk score calculated?", "Is Kestrion's cyber insurance still valid?"],
    "VENDOR": ["What evidence is still missing?", "Who has to approve it?", "Compare it with its peers"],
    "PORTFOLIO": ["Show the number of vendors in each risk band", "Which vendors have a knockout?", "Which vendors need approval from the CFO?"],
    "COMPARISON": ["Which of them has knockouts?", "Who has to approve the riskier one?"],
    "SYSTEM": ["What does the critic check?", "How was the agent evaluated?", "What are the risk band thresholds?"],
    "POLICY": ["What evidence does a tier 1 vendor need?", "Who approves a contract above INR 5 crore?"],
}


@dataclass
class ChatSession:
    """Conversation state the next turn needs: the vendors in focus and the last question (for follow-ups)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[dict] = field(default_factory=list)
    focus_vendor_ids: list[str] = field(default_factory=list)
    last_question: str | None = None
    last_route: str | None = None

    def to_state(self) -> dict:
        return {"focus_vendor_ids": self.focus_vendor_ids, "last_question": self.last_question, "last_route": self.last_route}

    @classmethod
    def from_state(cls, sid: str, state: dict | None, turns: list[dict] | None = None) -> "ChatSession":
        s = state or {}
        return cls(id=sid, turns=list(turns or []), focus_vendor_ids=list(s.get("focus_vendor_ids") or []),
                   last_question=s.get("last_question"), last_route=s.get("last_route"))


def max_steps() -> int:
    return int(os.getenv("CHAT_MAX_STEPS", "10"))


class _Budget(RuntimeError):
    pass


def _new_state(question: str) -> dict:
    return {"request": {"request_kind": "QUESTION", "question": question}, "steps": 0, "tokens": 0, "llm_calls": 0,
            "json_stats": {}, "tool_calls": 0, "retried": False}


def _step(state: dict, n: int = 1):
    if state["steps"] + n > max_steps():
        raise _Budget(f"step cap reached ({max_steps()} steps)")
    state["steps"] += n


def _tool(state: dict, ctx: ToolContext, name: str, args: dict):
    _step(state)
    state["tool_calls"] += 1
    return run_tool(name, args, ctx)


def _llm(state: dict, msgs: list[dict], model, tag: str, max_tokens: int, schema: dict | None = None):
    """One schema-validated LLM call through the agent's reliability layer (repair, one retry, budget)."""
    if state["steps"] + 1 > max_steps():                         # the chat's own cap (CHAT_MAX_STEPS) before the agent's
        raise _Budget(f"step cap reached ({max_steps()} steps)")
    out, upd = call_json(state, msgs, model, tag, max_tokens=max_tokens, schema=schema)
    for k in ("steps", "tokens", "llm_calls", "json_stats"):
        if k in upd:
            state[k] = upd[k]
    return out


def _audit(ctx: ToolContext, action: str, detail: dict):
    if ctx.audit:
        try:
            ctx.audit(action, detail)
        except Exception:
            pass


def _turn(route: str, behaviour: str, answer: str, method: str, **kw) -> dict:
    g = {"method": method, "ok": True, "problems": [], **(kw.pop("grounding", None) or {})}
    return {"kind": "chat", "route": route, "behaviour": behaviour, "answer": answer, "citations": kw.pop("citations", []),
            "table": kw.pop("table", None), "facts": kw.pop("facts", None), "grounding": g,
            "vendor_ids": kw.pop("vendor_ids", []), "suggestions": kw.pop("suggestions", SUGGEST.get(route, [])), **kw}


# ------------------------------------------------------------------ answer cache (knowledge routes only)
_cache_lock = threading.Lock()
_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _cache_key(*parts) -> str:
    return json.dumps([re.sub(r"\s+", " ", str(p or "")).strip().lower() for p in parts])


def _cache_get(key: str) -> dict | None:
    if os.getenv("CHAT_CACHE", "1") != "1":
        return None
    with _cache_lock:
        hit = _CACHE.get(key)
        if hit:
            _CACHE.move_to_end(key)
        return json.loads(json.dumps(hit)) if hit else None


def _cache_put(key: str, turn: dict):
    if os.getenv("CHAT_CACHE", "1") != "1":
        return
    with _cache_lock:
        _CACHE[key] = json.loads(json.dumps(turn, default=str))
        while len(_CACHE) > int(os.getenv("CHAT_CACHE_SIZE", "256")):
            _CACHE.popitem(last=False)


def clear_cache():
    with _cache_lock:
        _CACHE.clear()


# ------------------------------------------------------------------ knowledge (SYSTEM / POLICY)
def _corpus_passages(state: dict, ctx: ToolContext, query: str, question: str, notes: list[str]) -> list[dict]:
    """Internal policies and regulations (never vendor documents, e-mails or drafts) from document_search."""
    searches = [{"query": query[:500], "doc_types": RULE_DOC_TYPES, "include_superseded": bool(HISTORY_Q.search(question)), "k": 6}]
    if named_sources(question):
        searches.append({"query": query[:500], "doc_types": EXTERNAL_DOC_TYPES, "include_superseded": False, "k": 6})
    merged: dict[str, dict] = {}
    order: list[str] = []
    for args in searches:
        r = _tool(state, ctx, "document_search", args)
        if not r.ok:
            notes.append(f"policy and regulation search failed ({r.error}); answered from the project documents only")
            continue
        for p in r.data["passages"]:
            if p.get("vendor_id") or p["evidence_type"] in NON_EVIDENCE_TYPES:
                continue
            if p["doc_id"] in merged:
                if len(merged[p["doc_id"]]["text"].split()) < 450:
                    merged[p["doc_id"]]["text"] += "\n…\n" + p["text"]
                continue
            merged[p["doc_id"]] = {"id": p["doc_id"], "title": p["title"], "heading": p.get("heading") or "", "text": p["text"],
                                   "kind": "doc", "flags": p.get("flags") or [], "status": p.get("status"),
                                   "evidence_type": p["evidence_type"], "document_date": p.get("document_date")}
            order.append(p["doc_id"])
    out = [merged[d] for d in order]
    ranked, _ = order_by_named_source([{**p, "title": p["title"]} for p in out], question)
    return ranked


def _mix(kb: list[dict], corpus: list[dict], weights: tuple[float, float], k: int = 6) -> list[dict]:
    score: dict[str, float] = {}
    items: dict[str, dict] = {}
    for w, lst in ((weights[0], kb), (weights[1], corpus)):
        for rank, p in enumerate(lst, 1):
            items.setdefault(p["id"], p)
            score[p["id"]] = score.get(p["id"], 0.0) + w / (60 + rank)
    return [items[i] for i in sorted(items, key=lambda i: -score[i])][:k]


def _cite(p: dict, question: str) -> dict:
    snippet = grounding.plain(grounding.focus_window(question, p["text"], 60))
    return {"source_id": p["id"], "title": p.get("title"), "heading": p.get("heading") or None, "kind": p.get("kind", "doc"),
            "snippet": redact_pii(snippet)[0][:400]}


def _knowledge(q: str, d: router.Decision, session: ChatSession, ctx: ToolContext, state: dict) -> dict:
    context = None
    query = q
    if session.last_question and session.last_route in ("SYSTEM", "POLICY") and ELLIPTICAL.search(q):
        context = f"previous question: {session.last_question}"
        query = f"{session.last_question} {q}"
    key = _cache_key(d.route, query)
    hit = _cache_get(key)
    if hit:
        hit["grounding"]["cached"] = True
        return hit
    notes: list[str] = []
    query_x = expand_for(query, query)
    try:
        store = ctx.get_store()
    except Exception:
        store = None
    kb = knowledge.get_kb(store)
    _step(state)
    kb_passages = [{"id": h.chunk.chunk_id, "title": h.chunk.title, "heading": h.chunk.heading, "text": h.chunk.text,
                    "kind": "system" if h.chunk.doc_id == knowledge.LIVE_DOC else "doc", "flags": []}
                   for h in kb.search(query_x, 6)]
    try:
        corpus = _corpus_passages(state, ctx, query_x, query, notes)
    except _Budget:
        corpus = []
    passages = _mix(kb_passages, corpus, d.source_weights)
    cov = knowledge.coverage(query, [f"{p.get('title', '')} {p.get('heading', '')} {p['text']}" for p in passages])
    retrieval = {"kb_mode": kb.mode, "kb_note": kb.note or None, "coverage": round(cov, 2),
                 "candidates": [p["id"] for p in passages]}
    if not passages or cov < float(os.getenv("CHAT_MIN_COVERAGE", "0.5")):
        near = [p for p in passages[:2]] if cov >= 0.25 else []
        msg = ("I couldn't find this in the copilot's records (its documentation, policies, regulations or vendor "
               "data), so I won't guess." + (" The closest sources I found are listed below; they don't answer it directly."
                                              if near else ""))
        return _turn(d.route, "NOT_FOUND", msg, "abstained", citations=[_cite(p, q) for p in near], retrieval=retrieval,
                     grounding={"coverage": round(cov, 2), "note": f"best passage covers {cov:.0%} of the question's key words"})
    shown_all = [{**p, "text": grounding.focus_window(query, p["text"], 280)} for p in passages]
    out, shown, problems_first = None, shown_all, []
    for n in (6, 4, 3):
        shown = shown_all[:n]
        try:
            out = _llm(state, knowledge_messages(q, shown, context), ChatAnswer, "chat.answer", 450,
                       answer_schema([p["id"] for p in shown]))
            break
        except ContextTooLong:
            continue
        except (BudgetExceeded, _Budget) as e:
            notes.append(str(e))
            break
        except Exception as e:                                     # endpoint down, quota, 4xx: quote the sources instead
            notes.append(f"the model call failed ({type(e).__name__}); the answer quotes the sources instead")
            break
    texts = {p["id"]: p["text"] for p in shown}
    by_id = {p["id"]: p for p in shown}
    method, answer, cited, report, behaviour = "extractive", "", [], None, "ANSWER"
    if out is not None and not out.answerable:
        quote, ids = grounding.extractive(q, shown, 2)
        behaviour = "NOT_FOUND"
        answer = ("The sources I found don't answer this directly, so I won't guess."
                  + (" The closest passages:\n" + quote.split("\n", 1)[1] if quote else ""))
        cited, method = ids, "llm_abstained"
    elif out is not None:
        report = grounding.check(out.answer, out.citations, texts, q)
        if report.ok:
            answer, cited, method = out.answer, [c for c in out.citations if c in texts], "llm_checked"
        else:
            problems_first = report.problems
            state["retried"] = True
            out2 = None
            try:
                out2 = _llm(state, knowledge_messages(q, shown, context, feedback=report.problems), ChatAnswer, "chat.answer",
                            450, answer_schema(list(texts)))
            except Exception as e:                                 # noqa: BLE001  budget, context, endpoint: fall back below
                notes.append(f"retry skipped ({type(e).__name__})")
            if out2 is not None and out2.answerable:
                report = grounding.check(out2.answer, out2.citations, texts, q)
                if report.ok:
                    answer, cited, method = out2.answer, [c for c in out2.citations if c in texts], "llm_checked_after_retry"
                else:
                    out = out2
            if not answer:
                trimmed = grounding.trim(out.answer, out.citations, texts, q)
                if trimmed and grounding.check(trimmed, out.citations, texts, q).ok:
                    answer, cited, method = trimmed, [c for c in out.citations if c in texts], "llm_trimmed"
    if not answer:
        answer, cited = grounding.extractive(q, shown, 3)
        method = "extractive_after_failed_checks" if problems_first else "extractive"
        if not answer:
            return _turn(d.route, "NOT_FOUND", "The sources I found don't answer this, so I won't guess.", "abstained",
                         retrieval=retrieval, citations=[_cite(p, q) for p in shown[:2]])
    flagged = [p for p in shown if p["id"] in cited and any(str(f).startswith("possible_injection") for f in p.get("flags", []))]
    for p in flagged:
        notes.append(f"{p['title']} ({p['id']}) contains instruction-like text aimed at AI tools; it was treated as data.")
        _audit(ctx, "INJECTION_DETECTED", {"doc_id": p["id"], "source": "chat"})
    superseded = [p for p in shown if p["id"] in cited and p.get("status") == "SUPERSEDED"]
    for p in superseded:
        notes.append(f"{p['id']} is a superseded version, kept for audit only.")
    g = {"coverage": round(cov, 2), "retried": bool(problems_first), "first_attempt_problems": problems_first,
         "numbers_checked": report.numbers_checked if report else len(grounding.numbers(answer)),
         "sentences_checked": report.sentences_checked if report else 0, "note": "; ".join(notes) or None}
    shown_notes = [n for n in notes if "treated as data" in n or "superseded" in n or "model call failed" in n]
    turn = _turn(d.route, behaviour, answer + (("\n\n_" + " ".join(shown_notes) + "_") if shown_notes else ""),
                 method, citations=[_cite(by_id[c], q) for c in cited if c in by_id], retrieval=retrieval, grounding=g)
    if behaviour == "ANSWER" and method.startswith(("llm_checked", "extractive")) and not notes:
        _cache_put(key, turn)
    return turn


# ------------------------------------------------------------------ vendor
def _snapshot(comp: dict, risk: dict, as_of: str) -> str:
    weak = [f"{r['item']} {r['status']}" for r in comp["checklist"] if r["required"] and r["status"] in portfolio.WEAK]
    top = sorted(comp["findings"] + risk["findings"], key=lambda f: -f["points"])[:3]
    s = (f"{comp['vendor_id']} {comp['legal_name']} is tier {comp['tier']} and rated **{risk['band']}** "
         f"(overall score {risk['overall']}; the score alone gives {risk['raw_band']}) as of {as_of}.")
    if risk["knockouts"]:
        s += f" Knockouts: {', '.join(risk['knockouts'])}."
    s += f" Recommendation: {risk['recommendation']}."
    s += f" Evidence gaps: {', '.join(weak)}." if weak else " Every required evidence item is on file and current."
    if comp["contradictions"]:
        s += " CAIQ contradictions: " + "; ".join(comp["contradictions"]) + "."
    if top:
        s += " Main findings: " + "; ".join(f"{f['code']} ({f['severity']}): {f['statement']}" for f in top) + "."
    s += (f" Required approvers: {', '.join(risk['required_approvers'])}. No approval is recorded: only these approvers "
          "can decide, in the review case.")
    return s


def _facts_answer(comp: dict, kf: list[dict], as_of: str) -> str:
    return (f"From the vendor records for {comp['vendor_id']} {comp['legal_name']} (as of {as_of}):\n"
            + "\n".join(f"- **{k['subject']}**: {k['verdict']}" for k in kf))


def _vendor_citations(comp: dict, risk: dict, kf: list[dict] | None, passages: dict[str, dict] | None = None,
                      extra: list[dict] | None = None) -> list[dict]:
    titles: dict[str, str] = {}
    for f in comp["findings"] + risk["findings"]:
        for s in f["sources"]:
            titles.setdefault(s["doc_id"], s.get("title") or s["doc_id"])
    for r in comp["checklist"]:
        for s in r["sources"]:
            titles.setdefault(s["doc_id"], s.get("title") or s["doc_id"])
    titles.update({"POLICY-TPRM-001-v3": "Third-party risk management policy", "POLICY-DOA-002": "Delegation of authority policy"})
    ids: list[str] = []
    for c in extra or []:
        if c.get("doc_id") and c["doc_id"] not in ids:
            ids.append(c["doc_id"])
    for k in kf or []:
        ids += [d for d in k["doc_ids"] if d not in ids]
    if not ids:
        ids = ["POLICY-TPRM-001-v3", "POLICY-DOA-002"] + [s["doc_id"] for f in sorted(comp["findings"] + risk["findings"],
                                                                                     key=lambda f: -f["points"])[:3] for s in f["sources"][:1]]
    out = []
    for d in list(dict.fromkeys(ids))[:6]:
        p = (passages or {}).get(d)
        title = next((c.get("title") for c in extra or [] if c.get("doc_id") == d and c.get("title")), None) or titles.get(d) or (p or {}).get("title") or d
        snip = grounding.plain(redact_pii((p or {}).get("text", ""))[0])[:400]
        out.append({"source_id": d, "title": title, "kind": "doc", "snippet": snip or None})
    out.append({"source_id": "tool:compliance_checker+financial_risk_calculator", "title": "Rubric engine results", "kind": "tool"})
    return out


def _agent_grounded(st: dict, answer: str) -> list[str]:
    """Numbers in the agent's answer that are in none of its tool results, passages or the question (should be none)."""
    blob = json.dumps([r.get("data") for r in st.get("tool_results") or [] if r.get("ok")], default=str)
    blob += " " + " ".join(p.get("text", "") for p in st.get("passages") or []) + " " + (st["request"].get("question") or "")
    return sorted(grounding.numbers(answer) - grounding.numbers(blob) - grounding.SMALL)


def _vendor(q: str, d: router.Decision, session: ChatSession, ctx: ToolContext, as_of: str, state: dict) -> dict:
    vid = d.vendor_ids[0]
    comp_r = _tool(state, ctx, "compliance_checker", {"vendor_id": vid, "as_of": as_of})
    risk_r = _tool(state, ctx, "financial_risk_calculator", {"vendor_id": vid, "as_of": as_of})
    if not (comp_r.ok and risk_r.ok):
        err = comp_r.error or risk_r.error
        return _turn("VENDOR", "PARTIAL", f"I couldn't compute {vid}'s assessment because a tool failed ({err}). Nothing was "
                     "guessed; please try again, or open a full review.", "none", vendor_ids=[vid],
                     grounding={"ok": False, "problems": [f"tool failure: {err}"]})
    comp, risk = comp_r.data, risk_r.data
    kf = key_facts(q, comp, risk)
    facts = [{"subject": k["subject"], "verdict": k["verdict"], "doc_ids": k["doc_ids"]} for k in kf] or None
    base = dict(vendor_ids=[vid], facts=facts)
    if router.OVERVIEW.search(q) or os.getenv("CHAT_VENDOR_AGENT", "1") != "1":
        text = _snapshot(comp, risk, as_of) if (router.OVERVIEW.search(q) or not kf) else _facts_answer(comp, kf, as_of)
        return _turn("VENDOR", "ANSWER", text, "deterministic", citations=_vendor_citations(comp, risk, kf), **base,
                     grounding={"note": "written by code from compliance_checker and financial_risk_calculator"})
    from app.agents.graph import run_agent
    req = {"request_kind": "QUESTION", "vendor_ids": [vid], "question": q, "as_of": as_of, "requester_role": "procurement_analyst"}
    st, reason = None, ""
    try:
        with span("chat.vendor_agent", kind="span", input={"vendor_id": vid}):
            st = run_agent(req, ctx=ctx)
    except Exception as e:                                         # noqa: BLE001  (fallback below is the tested path)
        reason = f"the agent failed ({type(e).__name__}: {str(e)[:120]})"
        _audit(ctx, "CHAT_AGENT_ERROR", {"vendor_id": vid, "error": reason})
    if st is not None:
        for k in ("steps", "tokens", "llm_calls"):
            state[f"agent_{k}"] = st.get(k, 0)
        out = st.get("output") or {}
        b = out.get("behaviour")
        if b == "REFUSE":
            return _turn("REFUSE", "REFUSE", out.get("message", ""), "refusal", refusal_category=out.get("category"),
                         route_to=out.get("route_to"), **base)
        if b == "CLARIFY":
            return _turn("CLARIFY", "CLARIFY", out.get("question", ""), "none", clarify_options=out.get("options") or [], **base)
        critic = st.get("critic") or {}
        if b == "ANSWER" and out.get("answer"):
            loose = _agent_grounded(st, out["answer"])
            if critic.get("ok", True) and not loose:
                passages = {p["doc_id"]: p for p in st.get("passages") or []}
                cits = _vendor_citations(comp, risk, kf, passages, out.get("citations"))
                text = out["answer"] + ("\n\n**Key facts from the tools** (computed, not written by the model):\n"
                                        + "\n".join(f"- {k['subject']}: {k['verdict']}" for k in kf) if kf else
                                        f"\n\n**Status from the tools:** tier {comp['tier']}, band {risk['band']} (score "
                                        f"{risk['overall']}), recommendation {risk['recommendation']}"
                                        + (f", knockouts {', '.join(risk['knockouts'])}" if risk["knockouts"] else "") + ".")
                return _turn("VENDOR", "ANSWER", text, "agent_checked", citations=cits, **base,
                             grounding={"numbers_checked": len(grounding.numbers(out["answer"]) - grounding.SMALL),
                                        "note": "written by the agent's LLM writer; passed its code critic and the chat's number check",
                                        "agent_replanned": bool(st.get("replanned"))})
            reason = ("the agent's answer failed its code critic (" + "; ".join(critic.get("problems") or [])[:200] + ")"
                      if not critic.get("ok", True) else f"the agent's answer had numbers not found in its sources: {', '.join(loose[:4])}")
        else:
            reason = reason or ("the agent returned a partial answer: " + "; ".join(st.get("partial_reasons") or [])[:200])
    text = _facts_answer(comp, kf, as_of) if kf else _snapshot(comp, risk, as_of)
    return _turn("VENDOR", "ANSWER", text, "deterministic_fallback", citations=_vendor_citations(comp, risk, kf), **base,
                 grounding={"note": f"written by code from the tool results because {reason or 'the agent gave no answer'}"})


# ------------------------------------------------------------------ comparison and portfolio
def _comparison(d: router.Decision, ctx: ToolContext, as_of: str, state: dict) -> dict:
    r = _tool(state, ctx, "vendor_compare", {"vendor_ids": d.vendor_ids, "as_of": as_of})
    if not r.ok:
        return _turn("COMPARISON", "PARTIAL", f"I couldn't compare {', '.join(d.vendor_ids)} because vendor_compare failed "
                     f"({r.error}). Nothing was guessed.", "none", vendor_ids=d.vendor_ids,
                     grounding={"ok": False, "problems": [f"tool failure: {r.error}"]})
    c = r.data
    lines = [f"{row['rank']}. {row['vendor_id']} {row['legal_name']}: {row['band']} (score {row['overall']}), tier {row['tier']}, "
             f"{row['recommendation']}" + (f"; knockouts {', '.join(row['knockouts'])}" if row["knockouts"] else "")
             + (f"; top findings {', '.join(row['key_findings'])}" if row["key_findings"] else "") for row in c["rows"]]
    text = (f"Side by side, lower risk first (as of {as_of}):\n" + "\n".join(lines)
            + ("\n\n" + " ".join(c["caveats"]) if c["caveats"] else "")
            + "\n\nComputed by vendor_compare with the deterministic rubric; the ranking is advisory, a human decides.")
    table = [{k: (", ".join(v) if isinstance(v, list) else v) for k, v in row.items() if k != "dimension_points"} for row in c["rows"]]
    return _turn("COMPARISON", "ANSWER", text, "deterministic", vendor_ids=d.vendor_ids, table=table,
                 citations=[{"source_id": "POLICY-TPRM-001-v3", "title": "Third-party risk management policy (bands)", "kind": "doc"},
                            {"source_id": "tool:vendor_compare", "title": f"vendor_compare as of {as_of}", "kind": "tool"}],
                 grounding={"numbers_checked": len(grounding.numbers(text) - grounding.SMALL), "note": "written by code from vendor_compare"})


def _portfolio(q: str, d: router.Decision, ctx: ToolContext, as_of: str, state: dict) -> dict:
    try:
        rows = portfolio.rows(ctx, as_of, state)
    except Exception as e:                                         # noqa: BLE001
        return _turn("PORTFOLIO", "PARTIAL", f"I couldn't compute the portfolio because a tool failed ({str(e)[:200]}). "
                     "Nothing was guessed.", "none", grounding={"ok": False, "problems": [str(e)[:200]]})
    pq = portfolio.parse(q, d.vendor_ids)
    unmapped = [w for w in portfolio.unmapped_words(q)
                if not any(w in knowledge.content_terms(d_["legal_name"]) for d_ in router.directory(ctx.get_store()).values())]
    by = "rules"
    if not pq.has_filter() and unmapped:              # a condition the rules did not understand: never list everything
        groups = sorted({r["group"] for r in rows})
        try:
            got = _llm(state, portfolio_messages(q, portfolio.PortfolioQuery.model_json_schema(), groups),
                       portfolio.PortfolioQuery, "chat.portfolio", 300)
        except Exception:                                          # noqa: BLE001  no model: ask the user instead of guessing
            got = None
        if got is not None and got.has_filter():
            pq, by = got.model_copy(update={"vendor_ids": d.vendor_ids or got.vendor_ids}), "llm"
            unmapped = []
        else:
            return _turn("PORTFOLIO", "CLARIFY", "I couldn't map '" + " ".join(unmapped[:5]) + "' to the vendor fields I can "
                         "filter on, so I won't guess. I can filter by risk band, tier, review type (new vendor or renewal), "
                         "recommendation, knockouts, business-critical, evidence item and status (e.g. expired SOC 2), "
                         "required approvers and comparison group, and count, average, rank or break them down.",
                         "none", clarify_options=["Which vendors are rated HIGH?", "Which vendors have an expired SOC 2 report?",
                                                  "Show the number of vendors in each risk band"])
    res = portfolio.answer(pq, rows, as_of)
    if unmapped:                                      # filters applied, but some words were not: say so, don't hide it
        res["answer"] += (f"\n\nNot applied: I couldn't map '{' '.join(unmapped[:6])}' to a vendor field, so the result "
                          "uses only the filters listed above.")
    blob = json.dumps(rows, default=str)
    allowed = (grounding.numbers(blob) | grounding.numbers(q) | grounding.SMALL | grounding.numbers(" ".join(res["derived"]))
               | grounding.numbers(json.dumps(res["table"], default=str)))
    loose = sorted(grounding.numbers(res["answer"]) - allowed)          # code wrote the text: this is a self-check
    return _turn("PORTFOLIO", "ANSWER", res["answer"], "deterministic", table=res["table"], citations=res["citations"],
                 vendor_ids=res["matched"] if len(res["matched"]) <= 3 else [],
                 grounding={"ok": not loose, "problems": [f"unexpected numbers: {loose}"] if loose else [],
                            "numbers_checked": len(grounding.numbers(res["answer"]) - grounding.SMALL),
                            "note": f"filters read by {by}: {res['filters']}; written by code from the tool results"},
                 filters=res["filters"])


# ------------------------------------------------------------------ entry point
def _llm_router(state: dict):
    def route(q: str) -> str | None:
        if os.getenv("CHAT_LLM_ROUTER", os.getenv("AGENT_LLM_ROUTER", "1")) != "1":
            return None
        from app.agents import prompts as agent_prompts
        from app.agents import rules
        try:
            dec = _llm(state, agent_prompts.router_messages(q), RouteDecision, "chat.router", 60)
        except Exception:                                          # noqa: BLE001  the rules already ran; never block on this
            return None
        if dec and dec.refuse and dec.category:
            if dec.category == "APPROVAL_OR_COMMITMENT" and rules.status_question(q):
                return None
            return dec.category
        return None
    return route


def _dispatch(q: str, session: ChatSession, ctx: ToolContext, as_of: str, state: dict) -> dict:
    from app.agents import rules
    try:
        store = ctx.get_store()
    except Exception as e:
        return _turn("SYSTEM", "PARTIAL", f"The copilot's store is not available ({type(e).__name__}); run `make index` first. "
                     "Nothing was guessed.", "none", grounding={"ok": False, "problems": [str(e)[:200]]})
    d = router.decide(q, session.focus_vendor_ids, store, llm_router=_llm_router(state))
    if d.route == "HELP":
        t = _turn("HELP", "ANSWER", HELP_TEXT, "fixed_text")
    elif d.route == "REFUSE":
        t = _turn("REFUSE", "REFUSE", rules.MESSAGES[d.refusal_category], "refusal", refusal_category=d.refusal_category,
                  route_to=rules.ROUTE_TO.get(d.refusal_category), suggestions=[])
        _audit(ctx, "CHAT_REFUSAL", {"category": d.refusal_category, "by": d.by, "question": redact_pii(q)[0][:300],
                                     "session_id": session.id})
    elif d.route == "CLARIFY":
        t = _turn("CLARIFY", "CLARIFY", d.clarify["question"], "none", clarify_options=d.clarify.get("options") or [],
                  suggestions=["Which vendors are in the HRMS group?", "List all tier 1 vendors"])
    elif d.route == "VENDOR":
        t = _vendor(q, d, session, ctx, as_of, state)
    elif d.route == "COMPARISON":
        t = _comparison(d, ctx, as_of, state)
    elif d.route == "PORTFOLIO":
        t = _portfolio(q, d, ctx, as_of, state)
    else:
        t = _knowledge(q, d, session, ctx, state)
    if d.note and t["behaviour"] in ("ANSWER", "PARTIAL"):
        t["answer"] += "\n\n" + d.note
        _audit(ctx, "MIXED_REQUEST_APPROVAL_DECLINED", {"question": redact_pii(q)[0][:300], "session_id": session.id})
    t["router"] = {"route": d.route, "reason": d.reason, "by": d.by}
    return t


def ask(message: str, session: ChatSession | None = None, *, ctx: ToolContext | None = None, as_of: str | None = None,
        user: str | None = None) -> dict:
    """Answer one chat message. Never raises: failures become a PARTIAL turn that says what failed."""
    session = session if session is not None else ChatSession()
    ctx = ctx or ToolContext()
    as_of = str(as_of or DEFAULT_AS_OF)
    q = (message or "").strip()[:2000]
    state = _new_state(q)
    t0 = time.time()
    with start_trace("chat.turn", input=redact_pii(q)[0], metadata={"session_id": session.id}, user_id=user,
                     session_id=session.id) as tr:
        try:
            turn = _dispatch(q, session, ctx, as_of, state)
        except _Budget as e:
            turn = _turn("SYSTEM", "PARTIAL", f"I stopped before finishing ({e}). Nothing was guessed; try a narrower question.",
                         "none", grounding={"ok": False, "problems": [str(e)]})
        except Exception as e:                                     # noqa: BLE001
            turn = _turn("SYSTEM", "PARTIAL", f"Something went wrong while answering ({type(e).__name__}). Nothing was guessed.",
                         "none", grounding={"ok": False, "problems": [f"{type(e).__name__}: {str(e)[:200]}"]})
            _audit(ctx, "CHAT_ERROR", {"error": f"{type(e).__name__}: {str(e)[:300]}", "session_id": session.id})
        turn["answer"] = redact_pii(turn["answer"])[0]
        turn.update(steps=state["steps"], tool_calls=state["tool_calls"], llm_calls=state["llm_calls"] + state.get("agent_llm_calls", 0),
                    tokens=state["tokens"] + state.get("agent_tokens", 0), agent_steps=state.get("agent_steps"),
                    latency_ms=int((time.time() - t0) * 1000), trace_id=current_trace_id() or getattr(tr, "id", None),
                    as_of=as_of, session_id=session.id, json_stats=state.get("json_stats") or None)
    session.turns += [{"role": "user", "content": {"text": q}}, {"role": "assistant", "content": turn}]
    if turn["route"] in ("VENDOR", "COMPARISON") and turn.get("vendor_ids"):
        session.focus_vendor_ids = list(turn["vendor_ids"])
    elif turn["route"] == "PORTFOLIO" and len(turn.get("vendor_ids") or []) == 1:
        session.focus_vendor_ids = list(turn["vendor_ids"])
    if turn["behaviour"] != "REFUSE":
        session.last_question, session.last_route = q, turn["route"]
    return turn


def llm_backend() -> str:
    return llm_client.backend()
