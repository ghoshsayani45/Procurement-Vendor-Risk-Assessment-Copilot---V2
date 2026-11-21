"""Markdown export of a review (step 3.7): what the approvers read and file. Pure functions, no Streamlit, so they are
unit-tested and reused by the UI's download button.

    from app.ui.report_md import review_markdown, group_checklist
    text = review_markdown(review)          # review = GET /reviews/{id}
"""
from __future__ import annotations

STATUS_GROUPS = [("Verified", ("VERIFIED",)), ("Expiring soon", ("EXPIRING_SOON",)), ("Expired", ("EXPIRED",)),
                 ("Missing", ("MISSING",)), ("Contradictory or insufficient", ("CONTRADICTORY", "INSUFFICIENT")),
                 ("Not required", ("NOT_REQUIRED",))]


def group_checklist(checklist: list[dict]) -> list[tuple[str, list[dict]]]:
    """Checklist rows grouped by status, in the order a reviewer acts on them; unknown statuses go to 'Other'."""
    known = {s for _, ss in STATUS_GROUPS for s in ss}
    out = [(label, [r for r in checklist if r["status"] in ss]) for label, ss in STATUS_GROUPS]
    other = [r for r in checklist if r["status"] not in known]
    if other:
        out.append(("Other", other))
    return [(label, rows) for label, rows in out if rows]


def agent_turns(review: dict) -> list[dict]:
    return [t["content"] for t in review.get("turns", []) if t["role"] == "assistant" and t["content"].get("kind") == "answer"]


def latest_override(review: dict) -> dict | None:
    ovs = ((review.get("score_json") or {}).get("overrides")) or []
    return ovs[-1] if ovs else None


def _cell(v) -> str:
    return str(v if v not in (None, "") else "–").replace("|", "\\|").replace("\n", " ")


def review_markdown(review: dict) -> str:
    score = review.get("score_json") or {}
    comp, risk = score.get("compliance") or {}, score.get("risk") or {}
    vendor = review.get("vendor") or {}
    name = vendor.get("legal_name") or comp.get("legal_name") or risk.get("legal_name") or review.get("vendor_id")
    turns = agent_turns(review)
    first = turns[0] if turns else {}
    out = first.get("output") or {}
    L = [f"# Vendor risk review: {name}", "",
         f"Review `{review.get('id')}` · {review.get('request_kind')} · {review.get('review_type') or '–'} · as of "
         f"{review.get('as_of')} · status **{review.get('status')}** · requested by {review.get('requester')}", ""]
    if review.get("question"):
        L += [f"> {review['question']}", ""]
    L += ["*Prepared by the vendor-risk copilot. The copilot recommends; it never approves. Scores come from the "
          "deterministic rubric engine, not from the language model.*", ""]

    if out.get("behaviour") == "REFUSE":
        L += ["## Request refused", "", out.get("message") or out.get("answer") or "", "",
              f"Category: `{out.get('category')}`. Routed to: {out.get('route_to') or 'a human'}.", ""]
    if out.get("behaviour") == "CLARIFY":
        L += ["## Clarification needed", "", out.get("question", ""), ""]
    if out.get("behaviour") == "PARTIAL" or first.get("partial_reasons"):
        L += ["## Partial result", "", "The copilot could not complete every step, so this report is incomplete:", ""]
        L += [f"- {r}" for r in first.get("partial_reasons") or ["see the audit trail"]] + [""]

    if risk:
        L += ["## Risk score", "", f"**{risk['band']}** (score {risk['overall']}; score alone gives {risk['raw_band']}) · "
              f"rubric v{risk.get('rubric_version', '?')}", ""]
        ov = latest_override(review)
        if ov:
            L += [f"**Reviewer override:** {ov['reviewer_name']} set the band to **{ov['band']}** (computed: {ov['computed_band']}) "
                  f"on {ov['at']}. Reason: {ov['reason']}", ""]
        for k in risk.get("knockouts", []):
            L.append(f"- Knockout: {k}")
        L += ["", "| Dimension | Weight | Points | Weighted | Rules triggered |", "|---|---|---|---|---|"]
        for d in risk.get("dimensions", []):
            L.append(f"| {_cell(d['dimension'])} | {_cell(d.get('weight'))} | {_cell(d.get('points'))} | {_cell(d.get('weighted'))} | "
                     f"{_cell(', '.join(d.get('triggered_rules') or []))} |")
        L += ["", f"**Recommendation:** {risk['recommendation']}", ""]
        L += [f"- Condition: {c}" for c in risk.get("conditions", [])]
        L += ["", f"**Required approvers:** {', '.join(risk.get('required_approvers') or []) or '–'}", ""]

    if out.get("summary"):
        L += ["## Summary", "", out["summary"], ""]
    elif out.get("answer") and out.get("behaviour") in ("ANSWER", "PARTIAL"):
        L += ["## Answer", "", out["answer"], ""]

    if comp.get("checklist"):
        L += ["## Evidence checklist", "", f"Tier {comp.get('tier')}" + (f": {'; '.join(comp.get('tier_reasons', []))}" if comp.get("tier_reasons") else ""), ""]
        for label, rows in group_checklist(comp["checklist"]):
            L += [f"### {label} ({len(rows)})", ""]
            for r in rows:
                src = ", ".join(s["doc_id"] for s in r.get("sources", []))
                L.append(f"- **{r['item']}**{' (required)' if r.get('required') else ''}: {r.get('reason') or r['status']}"
                         + (f" [{src}]" if src else ""))
            L.append("")
        for w in comp.get("document_warnings") or []:
            L.append(f"- Possible prompt injection in `{w['doc_id']}` ({w.get('title')}): treated as data, not followed.")
        for c in comp.get("contradictions", []):
            L.append(f"- Contradiction: {c}")
        if comp.get("contradictions") or comp.get("document_warnings"):
            L.append("")

    findings = comp.get("findings", []) + risk.get("findings", [])
    if findings:
        L += ["## Findings", "", "| Code | Dimension | Severity | Points | Statement | Sources |", "|---|---|---|---|---|---|"]
        for f in findings:
            L.append(f"| {_cell(f['code'])} | {_cell(f['dimension'])} | {_cell(f['severity'])} | {_cell(f['points'])} | "
                     f"{_cell(f['statement'])} | {_cell(', '.join(s['doc_id'] for s in f.get('sources', [])))} |")
        L.append("")

    gaps = out.get("gaps_and_uncertainties") or []
    if gaps:
        L += ["## Gaps and uncertainties", ""] + [f"- {g}" for g in gaps] + [""]

    comparison = score.get("comparison") or review.get("comparison")
    if comparison:
        L += ["## Comparison", "", "| Rank | Vendor | Tier | Score | Band | Recommendation |", "|---|---|---|---|---|---|"]
        for r in comparison.get("rows", []):
            L.append(f"| {r.get('rank')} | {_cell(r.get('legal_name'))} ({r.get('vendor_id')}) | {_cell(r.get('tier'))} | "
                     f"{_cell(r.get('overall'))} | {_cell(r.get('band'))} | {_cell(r.get('recommendation'))} |")
        L += [""] + [f"- {c}" for c in comparison.get("caveats", [])] + [""]

    cites = [c for c in out.get("citations") or [] if not str(c.get("source_id", "")).startswith("tool:")]
    if cites:
        L += ["## Sources cited", ""]
        L += [f"- `{c.get('doc_id') or c.get('source_id')}` {c.get('title') or ''}"
              + (f" ({c['document_date']})" if c.get("document_date") else "") for c in cites] + [""]

    if len(turns) > 1:
        L += ["## Follow-up questions", ""]
        users = [t["content"].get("text") for t in review.get("turns", []) if t["role"] == "user"]
        for q, t in zip(users[1:] if review.get("question") else users, turns[1:]):
            o = t.get("output") or {}
            L += [f"**Q:** {q}", "", f"**A:** {o.get('answer') or o.get('message') or o.get('question') or ''}", ""]

    L += ["## Decision", ""]
    L += [f"Recorded: **{review['decision']}**." if review.get("decision") else
          "No decision recorded. Only the required approvers can decide, in the review case.", ""]
    L += ["## Audit trail", "", "| Time | Actor | Action |", "|---|---|---|"]
    L += [f"| {_cell((a.get('created_at') or '')[:19])} | {_cell(a.get('actor'))} | {_cell(a.get('action'))} |"
          for a in review.get("audit", [])]
    if first.get("trace_id") or review.get("trace_id"):
        L += ["", f"Trace: `{first.get('trace_id') or review.get('trace_id')}`"]
    return "\n".join(L) + "\n"
