"""Vendor Risk Copilot UI: submit a case, watch the agent's plan, check the evidence, read the scorecard,
override the band with a reason, record the human decision, export the report.

    streamlit run app/ui/streamlit_app.py

Data source (sidebar): the live API (API_URL + an API key) or the mock JSON in app/ui/mock_review.json (no backend;
regenerate with `python -m scripts.make_ui_mock`). The four demo paths are one click each in live mode; the
tool-failure path needs the API started with API_DEMO_FAULTS=1. The copilot never approves: decisions and overrides
are recorded by a human key only, and every one is an audit event.
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


with st.sidebar:
    st.header("Vendor Risk Copilot")
    source = st.radio("Data source", ["Live API", "Mock JSON"], index=0 if os.getenv("API_KEY") else 1, horizontal=True)
    review = None
    if source == "Live API":
        st.session_state.api_url = st.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
        st.session_state.api_key = st.text_input("API key (human reviewer)", os.getenv("API_KEY", ""), type="password")
        if not st.session_state.api_key:
            st.info("Enter an API key to start.")
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
        if MOCK.exists():
            review = json.loads(MOCK.read_text(encoding="utf-8"))
            st.caption("Mock mode: read-only. Decisions, overrides and follow-ups need the live API.")
        else:
            st.warning("No mock JSON: run `python -m scripts.make_ui_mock`.")

if st.session_state.get("flash"):
    st.error(st.session_state.pop("flash"))
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
        st.progress(min(1.0, (c.get("steps") or 0) / 10), text=f"{c.get('steps')} of 10 steps")
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
