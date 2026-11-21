"""fix for test failure pattern 1 ("the tool's verdict does not reach the answer").

The tools already compute the verdict a vendor question asks about (a checklist status, a finding, the band), but the 8B
writer often does not state it: it read "CYBER_INSURANCE: VERIFIED" (a certificate is on file) and missed the separate
finding that the cover is below the tier minimum. key_facts() picks, in code, the tool facts that match the question's
subject and joins the checklist row with its findings into one verdict line; the writer is told to lead with it.

Subjects are matched by domain vocabulary (evidence types, rubric dimensions), not by eval wording. Each line carries
the doc ids that evidence it, so the answer can cite them.

D43 (citation precision): a line's doc ids are only documents that support its verdict. A "missing" finding has no
evidence (the tool falls back to the vendor's security documents as its source), so it cites only the rule source; a
dimension's policy (contract or data-protection standard) is cited only when the question asks about the policy itself.
A question about what is missing gets one line listing the required items not on file, citing the rule source.
"""
from __future__ import annotations

import re

_I = re.I
# subject -> (question pattern, checklist item, finding-code fragments, rubric dimension)
SUBJECTS: list[tuple[str, re.Pattern, str | None, tuple[str, ...], str | None]] = [
    ("cyber insurance", re.compile(r"\binsur\w*|\bcover(age)?\b", _I), "CYBER_INSURANCE", ("INSURANCE",), None),
    ("SOC 2", re.compile(r"\bsoc\s*2|\bsoc2\b|\battestation\b", _I), "SOC2_TYPE2", ("SOC2",), None),
    ("ISO 27001", re.compile(r"\biso\b|27001", _I), "ISO27001", ("ISO27001",), None),
    ("CAIQ", re.compile(r"\bcaiq\b|questionnaire|\bencrypt\w*|\bclaims?\b|\bsays\b|self-assess", _I), "CAIQ", ("CAIQ",), None),
    ("penetration test", re.compile(r"\bpen(etration)?[\s-]?test|\bpentest", _I), "PENTEST", ("PENTEST",), None),
    ("BCP / DR test", re.compile(r"disaster recovery|\bdr\b|business continuity|\bbcp\b|recovery test", _I), "BCP_DR_TEST",
     ("BCP", "DR_TEST"), None),
    ("data processing agreement", re.compile(r"data processing agreement|\bdpa\b", _I), "DPA", ("DPA",), None),
    ("financial statements", re.compile(r"financ\w*|liquidity|solven\w*|current ratio|debt|revenue|loss\b", _I),
     "FINANCIALS", ("FIN-",), "financial_health"),
    ("contract terms", re.compile(r"contract (terms?|standards?|clauses?)|liabilit\w*|terminat\w*|auto-?renew\w*|notice period|"
                                  r"price escalation|audit rights?", _I), None, ("CON-",), "contract_terms"),
    ("data protection", re.compile(r"breach|data (residency|location|leaves?|stored)|offshore|sub-?processor|personal data", _I),
     None, ("DP-",), "data_protection"),
    ("service performance", re.compile(r"\bsla\b|uptime|availability|incident|service level|performance", _I), None,
     ("OPS-",), "operational_performance"),
]
RISK_Q = re.compile(r"\b(low|high|medium|critical)[\s-]risk\b|risk (band|score|rating|level)|\bband\b|how risky|overall risk|"
                    r"\brecommend\w*", _I)
POLICY_Q = re.compile(r"\bpolic\w*|\bstandards?\b|\bcomplian\w*|\b(allowed|acceptable|permitted|compliant)\b|"
                      r"\bmeets?\b|\brequire(d|ment)s?\b|\b(maximum|minimum|limit)\b", _I)
MISSING_Q = re.compile(r"\bnot (provided|submitted|supplied|on file)|\b(missing|outstanding|lacking)\b|"
                       r"\b(which|what) (required )?(evidence|documents?)\b.*\b(still|yet|need|lack|without)", _I)
GAP = ("MISSING", "EXPIRED", "CONTRADICTORY", "UNVERIFIED", "INSUFFICIENT")
RULE_SOURCE = "POLICY-TPRM-001-v3"
APPROVAL_Q = re.compile(r"\bapprov\w*|sign(ed)?[\s-]off|\bwho (must|needs to|has to) (approve|sign)", _I)


def _docs(*groups) -> list[str]:
    out: list[str] = []
    for g in groups:
        for s in g:
            if s and s not in out:
                out.append(s)
    return out[:6]


def key_facts(question: str, compliance: dict | None, risk: dict | None) -> list[dict]:
    """Tool facts that answer this question, as verdict lines with their doc ids (empty when nothing matches)."""
    q = question or ""
    comp, risk = compliance or {}, risk or {}
    findings = (comp.get("findings") or []) + (risk.get("findings") or [])
    out: list[dict] = []
    for name, pat, item, frags, dim in SUBJECTS:
        if not pat.search(q):
            continue
        row = next((r for r in comp.get("checklist") or [] if r["item"] == item), None) if item else None
        fs = [f for f in findings if any(f["code"].startswith(x) or f"-{x}" in f["code"] for x in frags)
              or (dim and f.get("dimension") == dim)]
        fs = list({f["code"]: f for f in fs}.values())
        parts = []
        if row:
            parts.append(f"{row['item']} {row['status']}"
                         + (" (required for this tier)" if row.get("required") else " (not required for this tier)")
                         + f": {row['reason']}")
        if fs:
            parts += [f"finding {f['code']} ({f['severity']}): {f['statement']}" for f in fs]
        elif dim:
            parts.append(f"no {name} findings: the rubric's {name} checks found nothing to flag")
        if row and row["status"] == "VERIFIED" and fs:
            parts.append("so the evidence is on file but does NOT meet the requirement")
        if not parts:
            continue
        asks_policy = bool(POLICY_Q.search(q))
        docs = _docs([s["doc_id"] for s in (row or {}).get("sources", [])],
                     [s["doc_id"] for f in fs if not f["code"].endswith("-MISSING") for s in f.get("sources", [])],
                     [RULE_SOURCE] if row else [],
                     ["POLICY-CS-003"] if dim == "contract_terms" and (asks_policy or not fs) else [],
                     ["POLICY-DPS-004"] if dim == "data_protection" and (asks_policy or not fs) else [])
        out.append({"subject": name, "verdict": "; ".join(parts), "doc_ids": docs})
    gaps = [r for r in comp.get("checklist") or [] if r.get("required") and str(r.get("status", "")).upper() in GAP]
    if comp and MISSING_Q.search(q) and not out:
        out.append({"subject": "required evidence not on file",
                    "verdict": ("; ".join(f"{r['item']} {r['status']}: {r['reason']}" for r in gaps)
                                if gaps else "every required evidence item is on file"),
                    "doc_ids": [RULE_SOURCE]})
    if risk and RISK_Q.search(q):
        out.append({"subject": "overall risk",
                    "verdict": (f"band {risk['band']} (score {risk['overall']}; the score alone gives {risk['raw_band']}"
                                + (f"; knockouts {', '.join(risk['knockouts'])}" if risk.get("knockouts") else "")
                                + f"); recommendation {risk['recommendation']}"),
                    "doc_ids": ["POLICY-TPRM-001-v3"]})
    if risk and APPROVAL_Q.search(q):
        out.append({"subject": "approval",
                    "verdict": ("no approval is recorded; an approval exists only when a required approver records it in the "
                                f"review case. Required approvers: {', '.join(risk['required_approvers'])}"),
                    "doc_ids": ["POLICY-DOA-002"]})
    return out[:5]
