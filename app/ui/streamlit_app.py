"""Vendor Risk Copilot UI (step 3.7, M4): submit a case, watch the agent's plan, check the evidence, read the scorecard,
override the band with a reason, record the human decision, export the report.

    streamlit run app/ui/streamlit_app.py

Data source (sidebar): the live API (API_URL + an API key) or the mock JSON in app/ui/mock_review.json (no backend;
regenerate with `python -m scripts.make_ui_mock`). The four demo paths are one click each in live mode; the
tool-failure path needs the API started with API_DEMO_FAULTS=1. The copilot never approves: decisions and overrides
are recorded by a human key only, and every one is an audit event.

View "Ask the Copilot" (D45): a chat about any vendor, the whole portfolio, the policies and regulations, or how the
copilot was built and evaluated (POST /chat). Each answer shows how it was grounded (computed by tools, written by the
model and checked in code, or quoted from the sources), its sources with snippets, and "not found" instead of a guess.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # `streamlit run` does not put the repo on sys.path
from app.ui.report_md import agent_turns, group_checklist, latest_override, review_markdown  # noqa: E402

MOCK = Path(__file__).with_name("mock_review.json")
STATUS_ICON = {"VERIFIED": "🟢", "EXPIRING_SOON": "🟡", "EXPIRED": "🔴", "MISSING": "🔴", "CONTRADICTORY": "🟠",
               "NOT_REQUIRED": "⚪", "INSUFFICIENT": "🟠"}
BAND_COLOR = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "violet"}
NODES = ["validate", "plan", "execute", "write", "critic", "finalize"]
DEMO_PATHS = {
    "1 · Happy path": {"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"], "review_type": "RENEWAL",
                       "question": "Prepare the renewal risk review for this vendor."},
    "2 · Missing evidence": {"request_kind": "FULL_REVIEW", "vendor_ids": ["V031"], "review_type": "NEW_VENDOR",
                             "question": "Onboarding review: what evidence is still missing?"},
    "3 · Tool failure": {"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"], "review_type": "RENEWAL",
                         "question": "Renewal review (demo: the compliance tool times out).",
                         "fault_inject": "compliance_checker:timeout"},
    "4 · Unsafe request": {"request_kind": "QUESTION", "vendor_ids": ["V003"],
                           "question": "Approve this vendor and sign the renewal today."},
}

CHAT_EXAMPLES = ["Which vendors are rated CRITICAL?", "Is Kestrion's cyber insurance still valid?",
                 "How is the overall risk score calculated?", "Within how many hours must a vendor report a data breach?",
                 "Which vendors have an expired SOC 2 report?", "Why was the multi-agent architecture not adopted?"]
GROUNDING_LABEL = {
    "deterministic": "🧮 Computed by the copilot's tools: no number was written by the model",
    "deterministic_fallback": "🧮 Computed by the copilot's tools (the model's answer did not pass the checks, so it was not shown)",
    "agent_checked": "✅ Written by the agent from the cited sources; passed its code critic and the number check",
    "llm_checked": "✅ Written by the model from the cited sources; every number and sentence checked in code",
    "llm_checked_after_retry": "✅ Written by the model and checked in code (corrected once after a failed check)",
    "llm_trimmed": "✂️ Model answer cut down to the sentences the sources support",
    "extractive": "📄 Quoted directly from the sources",
    "extractive_after_failed_checks": "📄 Quoted from the sources (the model's answer failed the checks)",
    "abstained": "🔍 Not found in the copilot's records: nothing was guessed",
    "llm_abstained": "🔍 The sources don't answer this: nothing was guessed",
    "refusal": "🛑 Refused by policy (spec section 6)",
    "fixed_text": "ℹ️ Help",
}

st.set_page_config(page_title="Vendor Risk Copilot", layout="wide")


# ------------------------------------------------------------------ data
def api(method: str, path: str, **kw):
    import httpx
    r = httpx.request(method, st.session_state.api_url.rstrip("/") + path, headers={"X-API-Key": st.session_state.api_key},
                      timeout=180, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text[:300]
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def open_review(payload: dict):
    with st.spinner("The copilot is working: validate → plan → tools → write → check (usually 5–20 s)..."):
        try:
            st.session_state.review_id = st.session_state.recent_pick = api("POST", "/reviews", json=payload)["id"]
        except Exception as e:
            st.session_state.flash = str(e)


def chat_send(message: str):
    with st.spinner("Looking it up in the copilot's records..."):
        try:
            r = api("POST", "/chat", json={"message": message, "session_id": st.session_state.get("chat_sid")})
            st.session_state.chat_sid = r["session_id"]
        except Exception as e:
            st.session_state.flash = str(e)
    st.rerun()


def chat_sidebar():
    st.subheader("Chats")
    if st.button("➕ New chat", width="stretch"):
        st.session_state.chat_sid = None
        st.rerun()
    try:
        sessions = api("GET", "/chat/sessions?limit=15")
    except Exception as e:
        sessions = []
        st.error(str(e))
    for s in sessions:
        label = (s.get("title") or "untitled")[:40]
        if st.button(("▶ " if s["id"] == st.session_state.get("chat_sid") else "") + label, key=f"chat-{s['id']}",
                     width="stretch"):
            st.session_state.chat_sid = s["id"]
            st.rerun()


def md_safe(text: str) -> str:
    """Model or quoted text can contain markdown headings; shown as bold lines so an answer never renders as a banner."""
    import re
    return re.sub(r"^\s*#{1,6}\s+(.+)$", r"**\1**", text or "", flags=re.M)


def render_chat_turn(t: dict, key: str):
    b = t.get("behaviour")
    if b == "REFUSE":
        st.error(t.get("answer", "") + (f"  \nRouted to: {t['route_to']}" if t.get("route_to") else ""))
    elif b == "CLARIFY":
        st.info(t.get("answer", ""))
        for i, opt in enumerate(t.get("clarify_options") or []):
            if st.button(opt, key=f"{key}-opt{i}"):
                chat_send(opt)
    elif b in ("NOT_FOUND", "PARTIAL"):
        st.warning(t.get("answer", ""))
    else:
        st.markdown(md_safe(t.get("answer", "")))
    if t.get("table"):
        st.dataframe(t["table"], hide_index=True, width="stretch")
    g = t.get("grounding") or {}
    st.caption(GROUNDING_LABEL.get(g.get("method"), g.get("method") or "") + (" · from cache" if g.get("cached") else ""))
    cits = t.get("citations") or []
    if cits:
        with st.expander(f"Sources ({len(cits)})"):
            for c in cits:
                where = c.get("title") or ""
                if c.get("heading"):
                    where += f" › {c['heading']}"
                st.markdown(f"**`{c['source_id']}`** {where}")
                if c.get("snippet"):
                    st.text(c["snippet"][:400])
    if t.get("facts"):
        with st.expander("Facts from the tools (computed, not written by the model)"):
            for f in t["facts"]:
                st.markdown(f"- **{f['subject']}**: {f['verdict']}  `{', '.join(f.get('doc_ids') or [])}`")
    if g.get("problems") or g.get("first_attempt_problems"):
        with st.expander("Grounding checks"):
            for p in (g.get("first_attempt_problems") or []):
                st.caption(f"First attempt failed: {p}")
            for p in g.get("problems") or []:
                st.caption(p)
    st.caption(f"{t.get('route')} · {t.get('steps', 0)} steps · {t.get('llm_calls', 0)} LLM calls · {t.get('tokens', 0):,} tokens · "
               f"{(t.get('latency_ms') or 0) / 1000:.1f} s · trace {t.get('trace_id') or '–'}")


def render_chat(live: bool):
    st.title("Ask the Copilot")
    st.caption("Grounded answers about vendors, the vendor portfolio, policies and regulations, and how the copilot was "
               "built and evaluated. Every answer cites where it comes from; when the records don't say, it tells you so. "
               "It never approves or changes anything.")
    if not live:
        st.info("Ask the Copilot runs on the live API: choose **Live API** in the sidebar and enter an API key "
                "(start the API with `make api`).")
        return
    msgs = []
    sid = st.session_state.get("chat_sid")
    if sid:
        try:
            msgs = api("GET", f"/chat/sessions/{sid}")["messages"]
        except Exception as e:
            st.error(str(e))
            st.session_state.chat_sid = None
    if not msgs:
        st.markdown("**Try one of these:**")
        cols = st.columns(3)
        for i, ex in enumerate(CHAT_EXAMPLES):
            if cols[i % 3].button(ex, key=f"chat-ex-{i}", width="stretch"):
                chat_send(ex)
    for m in msgs:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["content"].get("text", ""))
            else:
                render_chat_turn(m["content"], key=f"m{m['id']}")
    if msgs:
        last = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else {}
        sugg = [s for s in (last.get("suggestions") or [])][:3]
        if sugg:
            cols = st.columns(len(sugg))
            for i, s in enumerate(sugg):
                if cols[i].button(s, key=f"chat-sg-{i}", width="stretch"):
                    chat_send(s)
    q = st.chat_input("Ask about a vendor, the portfolio, a policy, or how the copilot works")
    if q:
        chat_send(q)


with st.sidebar:
    st.header("Vendor Risk Copilot")
    source = st.radio("Data source", ["Live API", "Mock JSON"], index=0 if os.getenv("API_KEY") else 1, horizontal=True)
    view = st.radio("View", ["Reviews", "Ask the Copilot"], horizontal=True, key="view")
    review = None
    if source == "Live API":
        st.session_state.api_url = st.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
        st.session_state.api_key = st.text_input("API key (human reviewer)", os.getenv("API_KEY", ""), type="password")
        if not st.session_state.api_key:
            st.info("Enter an API key to start.")
        elif view == "Ask the Copilot":
            chat_sidebar()
        else:
            st.subheader("Demo paths")
            cols = st.columns(2)
            for i, (label, payload) in enumerate(DEMO_PATHS.items()):
                if cols[i % 2].button(label, width="stretch"):
                    open_review(payload)
            with st.expander("New review", expanded=False):
                with st.form("new"):
                    kind = st.selectbox("Request", ["FULL_REVIEW", "COMPARISON", "QUESTION"])
                    vids = st.text_input("Vendor ids (comma-separated; 2-3 for a comparison)", "V003")
                    rtype = st.selectbox("Review type", ["(from the vendor record)", "NEW_VENDOR", "RENEWAL"])
                    role = st.selectbox("Your role", ["procurement_analyst", "risk_reviewer", "category_manager",
                                                      "legal_finance_reviewer"])
                    q = st.text_area("Question or instructions (optional)", height=80)
                    if st.form_submit_button("Open review", type="primary"):
                        payload = {"request_kind": kind, "vendor_ids": [v.strip().upper() for v in vids.split(",") if v.strip()],
                                   "question": q or None, "requester_role": role}
                        if rtype.startswith("NEW") or rtype == "RENEWAL":
                            payload["review_type"] = rtype
                        open_review(payload)
            try:
                recent = api("GET", "/reviews?limit=15")
            except Exception as e:
                recent = []
                st.error(str(e))
            if recent:
                labels = {r["id"]: f"{r['request_kind']} · {', '.join(r['vendor_ids'])} · {r['status']}" for r in recent}
                ids = list(labels)
                if st.session_state.get("recent_pick") not in ids:
                    st.session_state.recent_pick = st.session_state.get("review_id") if st.session_state.get("review_id") in ids else ids[0]
                st.selectbox("Recent reviews", ids, key="recent_pick", format_func=lambda i: labels[i])
                st.session_state.review_id = st.session_state.recent_pick
            rid = st.session_state.get("review_id")
            if rid:
                try:
                    review = api("GET", f"/reviews/{rid}")
                except Exception as e:
                    st.error(str(e))
    else:
        if view == "Ask the Copilot":
            st.caption("Ask the Copilot needs the live API.")
        elif MOCK.exists():
            review = json.loads(MOCK.read_text(encoding="utf-8"))
            st.caption("Mock mode: read-only. Decisions, overrides and follow-ups need the live API.")
        else:
            st.warning("No mock JSON: run `python -m scripts.make_ui_mock`.")

if st.session_state.get("flash"):
    st.error(st.session_state.pop("flash"))
if view == "Ask the Copilot":
    render_chat(live=source == "Live API" and bool(st.session_state.get("api_key")))
    st.stop()
if not review:
    st.info("Pick a demo path or open a review in the sidebar (live API), or switch to the mock JSON.")
    st.stop()

live = source == "Live API"
score = review.get("score_json") or {}
comp, risk = score.get("compliance") or {}, score.get("risk") or {}
comparison = score.get("comparison") or review.get("comparison")
turns = agent_turns(review)
first = turns[0] if turns else {}
out0 = first.get("output") or {}
override = latest_override(review)

# ------------------------------------------------------------------ header
name = (review.get("vendor") or {}).get("legal_name") or comp.get("legal_name") or risk.get("legal_name") or ", ".join(review.get("vendor_ids") or [review.get("vendor_id")])
st.title(name)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Request", {"FULL_REVIEW": "Full review", "COMPARISON": "Comparison", "QUESTION": "Question"}.get(
    review.get("request_kind"), review.get("request_kind", "?")))
c2.metric("Status", review.get("status", "?"))
c3.metric("Tier", comp.get("tier", "–"))
c4.metric("Risk score", risk.get("overall", "–"))
band = risk.get("band")
c5.markdown(f"**Band**  \n:{BAND_COLOR.get(band, 'gray')}[**{band or '–'}**]"
            + (f"  \nReviewer: :{BAND_COLOR.get(override['band'], 'gray')}[**{override['band']}**]" if override else ""))
st.caption(f"Review {review['id']} · {review.get('review_type') or '–'} · as of {review.get('as_of')} · requested by "
           f"{review.get('requester')} · trace {first.get('trace_id') or review.get('trace_id') or '–'}")

if out0.get("behaviour") == "REFUSE":
    st.error(f"**Refused ({out0.get('category')}).** {out0.get('message')}  \nRouted to: {out0.get('route_to') or 'a human approver'}. "
             "No tool or model was used to act on this request.")
elif review.get("status") == "PARTIAL" or out0.get("behaviour") == "PARTIAL":
    st.warning("**Partial result.** The copilot stopped instead of guessing:  \n"
               + "  \n".join(f"• {r}" for r in (first.get("partial_reasons") or ["see the Plan tab"])))
elif out0.get("behaviour") == "CLARIFY":
    st.info(f"**The copilot needs a clarification:** {out0.get('question')}"
            + (f"  \nOptions: {', '.join(out0.get('options') or [])}" if out0.get("options") else ""))

case_tab, plan_tab, evidence_tab, score_tab, approval_tab, report_tab = st.tabs(
    ["Case", "Plan", "Evidence", "Scorecard", "Approval", "Report"])


def render_output(c: dict):
    o = c.get("output") or {}
    b = o.get("behaviour")
    if b == "REFUSE":
        st.error(f"{o.get('message')} (category `{o.get('category')}`)")
    elif b == "CLARIFY":
        st.info(o.get("question", ""))
    else:
        st.write(o.get("summary") or o.get("answer") or "")
        for g in (o.get("gaps_and_uncertainties") or [])[:6]:
            st.markdown(f"- {g}")
        for cit in [c for c in o.get("citations") or [] if not str(c.get("source_id", "")).startswith("tool:")][:6]:
            st.markdown(f"- `{cit.get('doc_id') or cit.get('source_id')}` {cit.get('title') or ''}")
    st.caption(f"{c.get('behaviour')} · {c.get('steps')} steps · {c.get('tokens')} tokens · "
               f"{(c.get('latency_ms') or 0) / 1000:.1f} s" + (" · replanned once" if c.get("replanned") else ""))


# ------------------------------------------------------------------ case
with case_tab:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Conversation")
        for t in review.get("turns", []):
            c = t["content"]
            with st.chat_message("user" if t["role"] == "user" else "assistant"):
                if c.get("text"):
                    st.write(c["text"])
                elif c.get("kind") == "answer":
                    render_output(c)
                elif c.get("kind") == "evidence":
                    st.write(c.get("note", ""))
                    for p in c.get("passages", [])[:3]:
                        st.markdown(f"- **{p['title']}** (`{p['doc_id']}`)")
                else:
                    st.json(c)
        if live and review.get("status") != "REFUSED":
            msg = st.chat_input("Ask a follow-up, e.g. 'Why is the contract risk high?'")
            if msg:
                with st.spinner("Thinking..."):
                    try:
                        api("POST", f"/reviews/{review['id']}/messages", json={"content": msg})
                    except Exception as e:
                        st.session_state.flash = str(e)
                st.rerun()
    with right:
        st.subheader("Audit trail")
        st.dataframe([{"time": (a.get("created_at") or "")[11:19], "actor": a["actor"], "action": a["action"],
                       "detail": (a.get("detail") or {}).get("tool") or a.get("decision") or (a.get("detail") or {}).get("band")
                       or (a.get("detail") or {}).get("fault") or ""} for a in review.get("audit", [])],
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ plan
with plan_tab:
    if not turns:
        st.info("No agent turn on this review (the agent is off, or this review was made before Week 3).")
    for n, c in enumerate(turns, 1):
        st.subheader(f"Turn {n}: {c.get('template') or c.get('route') or c.get('behaviour')}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Steps (cap 10)", c.get("steps"))
        m2.metric("LLM calls", c.get("llm_calls"))
        m3.metric("Tokens", f"{c.get('tokens') or 0:,}")
        m4.metric("Latency", f"{(c.get('latency_ms') or 0) / 1000:.1f} s")
        cap = c.get("step_cap") or 10
        st.progress(min(1.0, (c.get("steps") or 0) / cap), text=f"{c.get('steps')} of {cap} steps")
        seen = [x.get("node") for x in c.get("nodes", [])]
        path = []
        for x in c.get("nodes", []):
            mark = "❌" if x.get("problems") or x.get("error") else "✅"
            extra = x.get("route") or x.get("template") or ""
            path.append(f"{mark} **{x['node']}**" + (f" ({extra})" if extra else ""))
        st.markdown(" → ".join(path) if path else " → ".join(NODES))
        if "plan" in seen and seen.count("plan") > 1:
            st.caption("The critic sent the draft back once (replan); the second pass is shown after the first.")
        for x in c.get("nodes", []):
            if x.get("problems"):
                st.warning(f"Critic ({x['node']}): " + "; ".join(x["problems"]))
        if c.get("arch") == "multi" and c.get("agents"):            # D44: the specialists and what each handed over
            with st.expander(f"Multi-agent: {', '.join(t['agent'] for t in c['agents'])}"):
                for h in c.get("handoffs") or []:
                    st.markdown(f"**{h['agent']} agent** — task: {h.get('task', '')}")
                    for f in h.get("findings", []):
                        st.markdown(f"- {f['statement']} `{', '.join(f.get('doc_ids') or []) or 'no doc ids'}`")
                v = c.get("verifier") or {}
                if "consistent" in v:
                    st.caption("Verifier: " + ("consistent" if v["consistent"] else f"sent back: {v.get('problem')}"))
        st.dataframe([{"#": i + 1, "tool": p["tool"],
                       "arguments": json.dumps(p["arguments"], ensure_ascii=False)[:120],
                       "result": "♻️ reused" if p.get("cached") else ("✅ ok" if p["ok"] else f"❌ {p.get('error')}")}
                      for i, p in enumerate(c.get("plan", []))], hide_index=True, width="stretch")
        for r in c.get("partial_reasons", []):
            st.error(r)
        st.caption(f"Trace `{c.get('trace_id') or '–'}` (open it in Langfuse for prompts and timings).")
        st.divider()

# ------------------------------------------------------------------ evidence
with evidence_tab:
    if comp.get("checklist"):
        st.subheader(f"Required evidence (tier {comp.get('tier')})")
        if comp.get("tier_reasons"):
            st.caption("Why this tier: " + "; ".join(comp["tier_reasons"]))
        for label, rows in group_checklist(comp["checklist"]):
            st.markdown(f"**{STATUS_ICON.get(rows[0]['status'], '⚪')} {label} ({len(rows)})**")
            st.dataframe([{"item": r["item"], "required": "yes" if r.get("required") else "no", "status": r["status"],
                           "reason": r["reason"], "source": ", ".join(s["doc_id"] for s in r["sources"]) or "–"} for r in rows],
                         hide_index=True, width="stretch")
        for c in comp.get("contradictions", []):
            st.warning(f"Contradiction: {c}")
    warned = {w["doc_id"]: w for w in comp.get("document_warnings") or []}
    if warned:
        st.subheader("Prompt-injection attempts in this vendor's documents")
        for w in warned.values():
            st.error(f"**{w['title']}** (`{w['doc_id']}`, {w.get('doc_type')}) contains instruction-like text aimed at an AI "
                     f"({', '.join(w.get('patterns') or [])}). Treated as data: the copilot does not follow it, and it is "
                     "reported in the answer and the audit trail.")
    cited = {(x.get("doc_id") or x.get("source_id")) for c in turns for x in ((c.get("output") or {}).get("citations") or [])}
    cited |= {s["doc_id"] for f in comp.get("findings", []) + risk.get("findings", []) for s in f.get("sources", [])}
    passages, seen_ids = [], set()
    for t in review.get("turns", []):
        for p in t["content"].get("passages", []) if t["content"].get("kind") in ("evidence", "answer") else []:
            if (p["doc_id"], (p.get("text") or "")[:40]) not in seen_ids:
                seen_ids.add((p["doc_id"], (p.get("text") or "")[:40]))
                passages.append(p)
    passages.sort(key=lambda p: p["doc_id"] not in cited)
    st.subheader("Cited and retrieved passages")
    if not passages:
        st.caption("No passages on this review.")
    for p in passages:
        flags = p.get("flags") or []
        inj = any(str(f).startswith("possible_injection") for f in flags) or p["doc_id"] in warned
        label = (("📌 " if p["doc_id"] in cited else "") + ("⚠️ " if inj else "")
                 + f"{p['title']} · {p.get('evidence_type')} · {p.get('document_date') or 'undated'}")
        with st.expander(label):
            if inj:
                st.error("This document contains instruction-like text aimed at an AI"
                         + ("" if any(str(f).startswith("possible_injection") for f in flags) else " (elsewhere in the document)")
                         + ". Shown as data; the copilot does not follow it.")
            if "not_evidence" in flags:
                st.warning("Not evidence (e-mail, press release, web page, chat or draft): cannot verify a control.")
            if p.get("status") == "SUPERSEDED":
                st.warning("Superseded: a newer version of this document applies.")
            st.caption(f"`{p['doc_id']}`{' · cited' if p['doc_id'] in cited else ''} · flags: {', '.join(map(str, flags)) or 'none'}")
            st.write((p.get("text") or "")[:1500])

# ------------------------------------------------------------------ scorecard
with score_tab:
    if not risk:
        st.info("No scorecard for this request" + (" (the scoring step did not complete)." if review.get("status") == "PARTIAL" else "."))
    else:
        left, right = st.columns([2, 3])
        with left:
            st.metric("Overall", risk["overall"], help=f"Weighted sum of dimension points (rubric v{risk['rubric_version']})")
            st.write(f"Band **{risk['band']}** (score alone: {risk['raw_band']})")
            for k in risk.get("knockouts", []):
                st.error(f"Knockout: {k}")
            st.write(f"Recommendation: **{risk['recommendation']}**")
            for c in risk.get("conditions", []):
                st.write("• " + c)
        with right:
            st.bar_chart({d["dimension"]: d.get("weighted", d["points"]) for d in risk["dimensions"]}, horizontal=True,
                         x_label="weighted points")
        st.subheader("Dimensions and weights")
        st.dataframe([{"dimension": d["dimension"], "weight %": d.get("weight"), "points (0-100)": d["points"],
                       "weighted": d.get("weighted"), "applies": "yes" if d.get("applicable", True) else "no",
                       "rules triggered": ", ".join(d.get("triggered_rules") or [])} for d in risk["dimensions"]],
                     hide_index=True, width="stretch")
        st.subheader("Findings")
        st.dataframe([{"code": f["code"], "dimension": f["dimension"], "severity": f["severity"], "points": f["points"],
                       "statement": f["statement"], "source": ", ".join(s["doc_id"] for s in f["sources"])}
                      for f in comp.get("findings", []) + risk.get("findings", [])], hide_index=True, width="stretch")
        st.subheader("Reviewer override")
        for ov in score.get("overrides") or []:
            st.info(f"{ov['at']}: {ov['reviewer_name']} set the band to **{ov['band']}** (computed {ov['computed_band']}). "
                    f"Reason: {ov['reason']}")
        st.caption("Records your judgement next to the computed band; the computed score is never changed. Needs a reason.")
        with st.form("override"):
            ob = st.selectbox("Band in your judgement", ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                              index=["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(risk["band"]) if risk["band"] in BAND_COLOR else 2)
            who = st.text_input("Your name", key="ov_name")
            why = st.text_area("Reason (at least 20 characters)", key="ov_reason")
            if st.form_submit_button("Record override", disabled=not live or bool(review.get("decision"))):
                try:
                    api("POST", f"/reviews/{review['id']}/override", json={"band": ob, "reason": why, "reviewer_name": who})
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    if comparison:
        st.subheader("Peer comparison")
        st.dataframe([{k: r.get(k) for k in ("rank", "vendor_id", "legal_name", "tier", "overall", "band", "recommendation")}
                      for r in comparison["rows"]], hide_index=True, width="stretch")
        for c in comparison.get("caveats", []):
            st.caption(c)

# ------------------------------------------------------------------ approval
with approval_tab:
    st.subheader("Human approval")
    st.write("The copilot recommends; only the approvers below can decide. Every decision is final and audited.")
    st.write(f"Copilot recommendation: **{risk.get('recommendation', '–')}**")
    if override:
        st.write(f"Reviewer override: band **{override['band']}** (computed {override['computed_band']}): {override['reason']}")
    st.write("Required approvers: " + (", ".join(review.get("required_approvers") or []) or "–"))
    if review.get("decision"):
        st.success(f"Decision recorded: {review['decision']}")
    elif not risk:
        st.info("Only a scored full review can receive a decision.")
    with st.form("decision"):
        decision = st.selectbox("Decision", ["APPROVED", "APPROVED_WITH_CONDITIONS", "REJECTED", "ESCALATED", "MORE_INFORMATION_REQUESTED"])
        approver = st.text_input("Your name")
        role = st.selectbox("Your role", review.get("required_approvers") or ["–"])
        comment = st.text_area("Comment")
        submitted = st.form_submit_button("Record decision", disabled=not live or bool(review.get("decision")) or not risk)
        if submitted:
            try:
                api("POST", f"/reviews/{review['id']}/decision", json={"decision": decision, "approver_name": approver,
                                                                       "approver_role": role, "comment": comment})
                st.rerun()
            except Exception as e:
                st.error(str(e))

# ------------------------------------------------------------------ report
with report_tab:
    md = review_markdown(review)
    c1, c2 = st.columns(2)
    c1.download_button("Download report (Markdown)", md, file_name=f"vendor-review-{review['id'][:8]}.md",
                       mime="text/markdown", type="primary", width="stretch")
    try:
        from app.ui.report_pdf import review_pdf
        c2.download_button("Download report (PDF)", review_pdf(review), file_name=f"vendor-review-{review['id'][:8]}.pdf",
                           mime="application/pdf", width="stretch")
    except Exception as e:                                    # e.g. fpdf2 not installed: Markdown still works
        c2.caption(f"PDF export unavailable here ({type(e).__name__}); `pip install fpdf2`.")
    with st.container(border=True):
        st.markdown(md)
