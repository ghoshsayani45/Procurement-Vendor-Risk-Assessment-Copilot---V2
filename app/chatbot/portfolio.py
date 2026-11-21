"""Portfolio questions: "which vendors ...", "how many ...", "average score of ...", "breakdown by band" (D45).

Every number comes from the deterministic tools: compliance_checker and financial_risk_calculator run once per vendor
(through run_tool, so each call is traced and audited) and the rows are cached per as-of date. The question is mapped
to a PortfolioQuery by rules (bands, tiers, review type, recommendation, knockouts, evidence item + status, approvers,
comparison group, aggregate); the LLM is asked to fill the same schema only when the rules find a portfolio question but
no condition they understand. The answer text is written by code from the filtered rows and states the filters it
applied, so a reader can see exactly how the question was read. No LLM writes portfolio numbers.
"""
from __future__ import annotations

import re
import statistics
import threading
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import DEFAULT_AS_OF, ToolContext, run_tool

BANDS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RECS = ["PROCEED", "PROCEED_WITH_CONDITIONS", "HOLD_PENDING_EVIDENCE", "ESCALATE_HIGH_RISK"]
WEAK = ["MISSING", "EXPIRED", "CONTRADICTORY", "EXPIRING_SOON"]
ITEMS = ["SOC2_TYPE2", "ISO27001", "SECURITY_ATTESTATION", "CAIQ", "PENTEST", "CYBER_INSURANCE", "BCP_DR_TEST", "DPA", "FINANCIALS"]
KNOCKOUTS = ["KO-DEBARRED", "KO-NO-SECURITY-ATTESTATION", "KO-PERSONAL-DATA-NO-DPA", "KO-CONFLICT-OF-INTEREST"]
APPROVERS = ["Category Manager", "Head of Procurement", "Finance Controller", "Chief Financial Officer", "Procurement Committee",
             "Third-Party Risk Committee", "Legal Counsel", "Data Protection Officer", "Ethics & Compliance Officer"]
_I = re.I


class EvidenceFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: Optional[Literal["SOC2_TYPE2", "ISO27001", "SECURITY_ATTESTATION", "CAIQ", "PENTEST", "CYBER_INSURANCE",
                           "BCP_DR_TEST", "DPA", "FINANCIALS"]] = None
    statuses: list[Literal["VERIFIED", "EXPIRING_SOON", "EXPIRED", "MISSING", "CONTRADICTORY"]] = Field(default_factory=list)


class PortfolioQuery(BaseModel):
    """What a portfolio question can ask. Also the JSON schema the LLM fills when the rules cannot."""
    model_config = ConfigDict(extra="forbid")
    bands: list[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = Field(default_factory=list)
    tiers: list[Literal[1, 2, 3]] = Field(default_factory=list)
    review_types: list[Literal["NEW_VENDOR", "RENEWAL"]] = Field(default_factory=list)
    recommendations: list[Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "HOLD_PENDING_EVIDENCE", "ESCALATE_HIGH_RISK"]] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list, description="comparison group ids, e.g. G12-PAYMENTS")
    has_knockout: Optional[bool] = None
    business_critical: Optional[bool] = None
    knockouts: list[Literal["KO-DEBARRED", "KO-NO-SECURITY-ATTESTATION", "KO-PERSONAL-DATA-NO-DPA", "KO-CONFLICT-OF-INTEREST"]] = Field(default_factory=list)
    evidence: list[EvidenceFilter] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list, description="rubric finding codes or code prefixes, e.g. DP-BREACH-NOTICE-LONG, CON-")
    approvers: list[str] = Field(default_factory=list)
    vendor_ids: list[str] = Field(default_factory=list)
    aggregate: Literal["list", "count", "avg", "max", "min", "group_by"] = "list"
    group_by: Optional[Literal["band", "tier", "recommendation", "group", "review_type"]] = None
    top_n: Optional[int] = Field(default=None, ge=1, le=35)

    def has_filter(self) -> bool:
        return any([self.bands, self.tiers, self.review_types, self.recommendations, self.groups, self.has_knockout is not None,
                    self.business_critical is not None, self.knockouts, self.evidence, self.findings, self.approvers,
                    self.vendor_ids])

    def is_trivial(self) -> bool:
        return not self.has_filter() and self.aggregate == "list"


# ------------------------------------------------------------------ rows (tools, cached per as_of)
_lock = threading.Lock()
_ROWS: dict[tuple, list[dict]] = {}


def rows(ctx: ToolContext, as_of: str | None = None, steps: dict | None = None) -> list[dict]:
    """One row per vendor from vendor_search + compliance_checker + financial_risk_calculator. Raises on tool failure."""
    as_of = str(as_of or DEFAULT_AS_OF)
    key = (id(ctx.get_store()), as_of)
    with _lock:
        if key in _ROWS:
            return _ROWS[key]
    recs = ctx.get_store().get_vendor_records()
    out = []
    for vid in sorted(recs):
        rec = recs[vid]
        comp = run_tool("compliance_checker", {"vendor_id": vid, "as_of": as_of}, ctx)
        risk = run_tool("financial_risk_calculator", {"vendor_id": vid, "as_of": as_of}, ctx)
        if steps is not None:
            steps["tool_calls"] = steps.get("tool_calls", 0) + 2
        if not (comp.ok and risk.ok):
            raise RuntimeError(f"{vid}: {(comp.error or risk.error)}")
        c, r = comp.data, risk.data
        out.append({"vendor_id": vid, "legal_name": rec["legal_name"], "category": rec["category"],
                    "group": rec["comparison_group"], "review_type": rec["review_type"], "tier": c["tier"],
                    "band": r["band"], "raw_band": r["raw_band"], "overall": r["overall"],
                    "recommendation": r["recommendation"], "knockouts": r["knockouts"],
                    "approvers": r["required_approvers"], "annual_value_inr": rec["annual_value_inr"],
                    "business_critical": bool(rec.get("business_critical")),
                    "checklist": {x["item"]: x["status"] for x in c["checklist"]},
                    "evidence_sources": {x["item"]: [s["doc_id"] for s in x["sources"]] for x in c["checklist"]},
                    "findings": sorted(f["code"] for f in c["findings"] + r["findings"])})
    with _lock:
        _ROWS[key] = out
    return out


def reset():
    with _lock:
        _ROWS.clear()


# ------------------------------------------------------------------ question -> query (rules)
ITEM_WORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsoc\s*-?\s*2\b|\bsoc2\b", _I), "SOC2_TYPE2"),
    (re.compile(r"\biso\s*(27001)?\b|\b27001\b", _I), "ISO27001"),
    (re.compile(r"\bsecurity attestations?\b|\battestations?\b", _I), "SECURITY_ATTESTATION"),
    (re.compile(r"\bcaiq\b|\bquestionnaires?\b", _I), "CAIQ"),
    (re.compile(r"\bpen(etration)?[\s-]?tests?\b|\bpentests?\b", _I), "PENTEST"),
    (re.compile(r"\b(cyber[\s-]?)?insurance\b", _I), "CYBER_INSURANCE"),
    (re.compile(r"\b(bcp|dr test|disaster recovery tests?|recovery tests?|business continuity)\b", _I), "BCP_DR_TEST"),
    (re.compile(r"\bdpas?\b|\bdata processing agreements?\b", _I), "DPA"),
    (re.compile(r"\bfinancial statements?\b|\bfinancials\b|\baudited accounts\b", _I), "FINANCIALS"),
]
STATUS_WORDS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\bexpiring\b|\babout to expire\b|\bexpire soon\b", _I), ["EXPIRING_SOON"]),
    (re.compile(r"\bexpired\b|\bout of date\b|\blapsed\b", _I), ["EXPIRED"]),
    (re.compile(r"\bmissing\b|\bnot on file\b|\bno\b[^.?]{0,40}\bon file\b|\bwithout\b|\black(s|ing)?\b|\bno (valid )?(soc|iso|caiq|pen|cyber|insurance|dpa|bcp|dr|financ)", _I), ["MISSING"]),
    (re.compile(r"\bcontradict\w*\b|\binconsisten\w*\b", _I), ["CONTRADICTORY"]),
    (re.compile(r"\b(valid|verified|current|in date)\b", _I), ["VERIFIED"]),
    (re.compile(r"\b(gaps?|problems?|issues?|not verified|unverified|weak)\b", _I), list(WEAK)),
]
GROUP_WORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bhrms\b|\bpayroll\b|\bhr (software|systems?|saas)\b", _I), "G01-HRMS"),
    (re.compile(r"\bcloud hosting\b|\biaas\b|\bhosting\b", _I), "G02-CLOUD"),
    (re.compile(r"\bmssp\b|\bmanaged security\b|\bsecurity operations\b|\bsoc services\b", _I), "G03-MSSP"),
    (re.compile(r"\bcrm\b", _I), "G04-CRM"),
    (re.compile(r"\banalytics\b|\bbusiness intelligence\b|\bbi\b", _I), "G05-ANALYTICS"),
    (re.compile(r"\bit staffing\b|\bstaffing\b|\bmanaged services\b|\bit services\b", _I), "G06-ITSERV"),
    (re.compile(r"\bnetwork(ing)?\b", _I), "G07-NETWORK"),
    (re.compile(r"\bendpoint\b", _I), "G08-ENDPOINT"),
    (re.compile(r"\bai document\b|\bdocument (processing|intelligence)\b", _I), "G09-AIDOC"),
    (re.compile(r"\bbackup\b|\bdisaster recovery (vendors?|providers?|services?)\b", _I), "G10-BACKUP"),
    (re.compile(r"\bcollaboration\b|\bemail (saas|providers?|vendors?)\b", _I), "G11-COLLAB"),
    (re.compile(r"\bpayments?( gateways?)?\b", _I), "G12-PAYMENTS"),
    (re.compile(r"\btelecom\b|\bconnectivity\b", _I), "G13-TELECOM"),
    (re.compile(r"\bsoftware development\b|\bdev(elopment)? outsourcing\b", _I), "G14-DEVOUT"),
    (re.compile(r"\be-?learning\b|\blearning\b", _I), "G15-ELEARN"),
    (re.compile(r"\bbpo\b|\bcustomer support\b|\bcontact cent(re|er)s?\b|\bcall cent(re|er)s?\b", _I), "G16-BPO"),
    (re.compile(r"\bidentity\b|\bsso\b|\biam\b", _I), "G17-IAM"),
    (re.compile(r"\bcctv\b|\bfacility\b|\biot\b|\bsurveillance\b", _I), "G18-FACILITY"),
]
FINDING_WORDS: list[tuple[re.Pattern, list[str]]] = [          # rubric finding codes (app/scoring/rubric.yaml)
    (re.compile(r"\bbreach[\s-]?notif\w*|\bbreach[\s-]notice\b|\bbreach clause\b", _I), ["DP-BREACH-NOTICE-LONG", "DP-BREACH-NOTICE-ABSENT"]),
    (re.compile(r"\bsub[\s-]?processors?\b", _I), ["DP-NO-SUBPROCESSOR-CONTROL"]),
    (re.compile(r"\boffshore\b|\bcross[\s-]border\b|\bdata residency\b", _I), ["DP-OFFSHORE-NO-SAFEGUARD"]),
    (re.compile(r"\buncapped\b", _I), ["CON-CUSTOMER-UNCAPPED"]),
    (re.compile(r"\bliability caps?\b|\bvendor caps?\b|\blow cap\b", _I), ["CON-LOW-VENDOR-CAP", "CON-CUSTOMER-UNCAPPED"]),
    (re.compile(r"\bauto[\s-]?renew\w*|\bnotice period\b|\brenewal notice\b", _I), ["CON-AUTO-RENEW-LONG-NOTICE"]),
    (re.compile(r"\baudit rights?\b", _I), ["CON-NO-AUDIT-RIGHTS"]),
    (re.compile(r"\bterminat\w* for convenience\b|\btermination rights?\b", _I), ["CON-NO-TFC"]),
    (re.compile(r"\bprice (escalation|increases?|rises?)\b", _I), ["CON-PRICE-ESCALATION"]),
    (re.compile(r"\bcontract (findings?|issues?|problems?|risks?)\b", _I), ["CON-"]),
    (re.compile(r"\bdata[\s-]protection (findings?|issues?|problems?|risks?)\b", _I), ["DP-"]),
    (re.compile(r"\bliquidity\b|\bcurrent ratio\b", _I), ["FIN-LOW-LIQUIDITY", "FIN-WEAK-LIQUIDITY"]),
    (re.compile(r"\bleverage\b|\bdebt[\s-]to[\s-]equity\b|\bdebt\b", _I), ["FIN-HIGH-LEVERAGE", "FIN-ELEVATED-LEVERAGE"]),
    (re.compile(r"\boperating loss\w*|\bloss[\s-]making\b|\bnegative margin\b|\blosing money\b", _I), ["FIN-LOSS-MAKING"]),
    (re.compile(r"\brevenue (decline|fell|drop\w*|down)\b|\bdeclining revenue\b|\bfalling revenue\b", _I), ["FIN-REVENUE-DECLINE"]),
    (re.compile(r"\bfinancial (findings?|issues?|problems?|risks?|health)\b", _I), ["FIN-"]),
    (re.compile(r"\bsla\b|\bservice levels?\b|\buptime\b", _I), ["OPS-SLA-MISS-MAJOR", "OPS-SLA-MISS-MINOR"]),
    (re.compile(r"\bp1\b|\bincidents?\b|\boutages?\b", _I), ["OPS-P1-INCIDENTS", "OPS-P1-SOME"]),
    (re.compile(r"\blate deliver\w*|\bdelivery delays?\b", _I), ["OPS-LATE-DELIVERY"]),
    (re.compile(r"\bdisputes?\b", _I), ["OPS-OPEN-DISPUTE"]),
    (re.compile(r"\b(open\s+)?critical\s+(pen[\s-]?test\s+|penetration[\s-]test\s+)?findings?\b|"
                r"\bopen\s+critical\b(\s+(pen[\s-]?test|penetration[\s-]test))?(\s+findings?)?", _I), ["SEC-PENTEST-CRITICAL-OPEN"]),
    (re.compile(r"\b(insurance|cover(age)?) (below|under|too low|insufficient)\b|\bunder[\s-]?insured\b|\blow (insurance|cover)\b", _I), ["SEC-INSURANCE-LOW"]),
]
APPROVER_WORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcfo\b|\bchief financial officer\b", _I), "Chief Financial Officer"),
    (re.compile(r"\bprocurement committee\b", _I), "Procurement Committee"),
    (re.compile(r"\bhead of procurement\b", _I), "Head of Procurement"),
    (re.compile(r"\bfinance controller\b", _I), "Finance Controller"),
    (re.compile(r"\bcategory manager\b", _I), "Category Manager"),
    (re.compile(r"\blegal( counsel)?\b", _I), "Legal Counsel"),
    (re.compile(r"\bdpo\b|\bdata protection officer\b", _I), "Data Protection Officer"),
    (re.compile(r"\b(third[\s-]party )?risk committee\b|\btprc\b", _I), "Third-Party Risk Committee"),
    (re.compile(r"\bethics\b", _I), "Ethics & Compliance Officer"),
]
INTENT = re.compile(
    r"\b(which|what|how many|list|show|count|name|any|all)\b[^.?]{0,50}\bvendors\b|"
    r"\bvendors\b[^.?]{0,12}\b(with|that|whose|in|on|rated|where|having|without|needing|need|are|have|is)\b|"
    r"\bportfolio\b|\bacross (all |the )?vendors\b|\b(each|every|per) (risk )?(band|tier)\b|"
    r"\b(breakdown|distribution)\b[^.?]{0,40}\b(vendors?|bands?|tiers?|recommendations?)\b|"
    r"\bwhich vendor (has|is|was|needs?)\b[^.?]{0,40}\b(highest|lowest|most|least|riskiest|safest|best|worst|top)\b|"
    r"\b(average|mean|median) (overall )?(risk )?score\b|\bhow many\b(\s+[\w-]+){0,3}?\s+vendors\b", _I)
APPROVER_CONTEXT = re.compile(r"\bapprov\w*|\bsign[\s-]?off|\bsign\b", _I)


def looks_like_portfolio(q: str) -> bool:
    return bool(INTENT.search(q or ""))


def parse(question: str, vendor_ids: list[str] | None = None) -> PortfolioQuery:
    q = question or ""
    low = q.lower()
    pq = PortfolioQuery(vendor_ids=list(vendor_ids or []))
    for b in BANDS:                     # \b keeps "highest" / "lower" out; "business-critical" is not a band
        if re.search(rf"(?<!business[\s-])\b{b.lower()}\b", low):
            pq.bands.append(b)
    pq.tiers = sorted({int(t) for t in re.findall(r"\btier[\s-]?([123])\b", low)})        # type: ignore[assignment]
    if re.search(r"\brenewals?\b", low):
        pq.review_types.append("RENEWAL")
    if re.search(r"\bnew vendors?\b|\bonboarding\b|\bnew suppliers?\b", low):
        pq.review_types.append("NEW_VENDOR")
    if re.search(r"\bwith conditions\b|\bconditional(ly)?\b", low):
        pq.recommendations.append("PROCEED_WITH_CONDITIONS")
    elif re.search(r"\bproceed\b", low) or re.search(r"\bwithout conditions\b|\bunconditional", low):
        pq.recommendations.append("PROCEED")
    if re.search(r"\bon hold\b|\bhold pending\b|\bpending evidence\b|\bheld\b", low):
        pq.recommendations.append("HOLD_PENDING_EVIDENCE")
    if re.search(r"\bescalat\w*\b", low):
        pq.recommendations.append("ESCALATE_HIGH_RISK")
    if re.search(r"\bknock[\s-]?outs?\b", low):
        pq.has_knockout = True
    if re.search(r"\bbusiness[\s-]critical\b", low):
        pq.business_critical = True
    if re.search(r"\bdebar\w*\b", low):
        pq.knockouts.append("KO-DEBARRED")
    if re.search(r"\bconflicts? of interest\b|\bcoi\b", low):
        pq.knockouts.append("KO-CONFLICT-OF-INTEREST")
    rest = q
    for rx, codes in FINDING_WORDS:                # findings first: "critical pentest findings" is a finding, not evidence
        if rx.search(rest):
            pq.findings += [c for c in codes if c not in pq.findings]
            rest = rx.sub(" ", rest)
    items = []
    for rx, item in ITEM_WORDS:
        if rx.search(rest):
            items.append(item)
            rest = rx.sub(" ", rest)
    statuses: list[str] = []
    for rx, sts in STATUS_WORDS:
        if rx.search(q):
            statuses += [s for s in sts if s not in statuses]
    if items:                                      # no status word: show the item's status where it is required
        for it in items:
            pq.evidence.append(EvidenceFilter(item=it, statuses=statuses))                        # type: ignore[arg-type]
    elif statuses and re.search(r"\bevidence\b|\bdocuments?\b|\bcertificat\w*\b|\bchecklist\b", low):
        pq.evidence.append(EvidenceFilter(item=None, statuses=statuses))                          # type: ignore[arg-type]
    if "SEC-PENTEST-CRITICAL-OPEN" in pq.findings:          # "critical findings" is not the CRITICAL band
        pq.bands = [b for b in pq.bands if b != "CRITICAL"] if not re.search(r"\b(rated|band)\s+critical\b|\bcritical\s+(risk|band)\b", low) else pq.bands
    for rx, g in GROUP_WORDS:
        if rx.search(rest):
            pq.groups.append(g)
    if APPROVER_CONTEXT.search(q) or re.search(r"\bcfo\b|\bdpo\b", low):
        for rx, a in APPROVER_WORDS:
            if rx.search(q) and a not in pq.approvers:
                pq.approvers.append(a)
    if re.search(r"\b(each|every|per|by) (risk )?band\b|\bbands?\b[^.?]{0,20}\b(breakdown|distribution)\b|\b(breakdown|distribution)\b[^.?]{0,25}\bbands?\b", low):
        pq.aggregate, pq.group_by = "group_by", "band"
    elif re.search(r"\b(each|every|per|by) tier\b|\btiers?\b[^.?]{0,20}\b(breakdown|distribution)\b|\b(breakdown|distribution)\b[^.?]{0,25}\btiers?\b", low):
        pq.aggregate, pq.group_by = "group_by", "tier"
    elif re.search(r"\b(each|every|per|by) recommendation\b|\brecommendations?\b[^.?]{0,20}\b(breakdown|distribution)\b", low):
        pq.aggregate, pq.group_by = "group_by", "recommendation"
    elif re.search(r"\b(each|every|per|by) (comparison )?(group|category)\b", low):
        pq.aggregate, pq.group_by = "group_by", "group"
    elif re.search(r"\b(average|mean)\b", low):
        pq.aggregate = "avg"
    elif re.search(r"\b(highest|riskiest|most risky|worst|largest)\b|\btop\s*\d+\b", low):
        pq.aggregate = "max"
    elif re.search(r"\b(lowest|least risky|safest|best|smallest)\b", low):
        pq.aggregate = "min"
    elif re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", low):
        pq.aggregate = "count"
    m = re.search(r"\btop\s*(\d+)\b", low)
    if m:
        pq.top_n = max(1, min(35, int(m.group(1))))
    elif pq.aggregate in ("max", "min") and re.search(r"\bwhich vendor\b|\bthe vendor\b|\bwho\b", low):
        pq.top_n = 1
    return pq


def unmapped_words(question: str) -> list[str]:
    """Content words the rules could not place, so the caller can tell when a condition was not understood."""
    from app.chatbot.knowledge import content_terms
    known = content_terms("""which what how many list show count name any all vendors vendor supplier suppliers rated rating
    risk band bands tier tiers score scores overall have has having need needs needed with without that whose are is in on
    the of for by each every per breakdown distribution average mean median highest lowest top most least riskiest safest
    best worst number renewal renewals new onboarding recommended recommendation proceed conditions condition hold pending
    escalate escalated knockout knockouts debarred debarment conflict interest approval approve approvers approver sign
    off evidence document documents certificate checklist missing expired expiring valid verified current gaps problems
    issues file report reports soc iso caiq pentest penetration test insurance cyber dpa financial statements cfo dpo
    legal counsel committee procurement head finance controller category manager officer ethics compliance portfolio
    across currently still yet status statuses status' total overall group groups payment payments gateway gateways
    hosting cloud crm analytics staffing network endpoint backup telecom identity sso bpo cctv facility learning
    chief business critical breach notification notice clause clauses long sub processor processors offshore
    cross border residency uncapped liability cap caps auto renew renews renewal notice period audit rights termination
    convenience price escalation increase increases rise rises contract contracts protection liquidity ratio leverage
    debt equity operating loss making margin negative revenue decline declining falling fell drop dropping health sla
    service level levels uptime p1 incident incidents outage outages late delivery deliveries delays delay dispute
    disputes open finding findings coverage below under insured insufficient money losing type ii statement
    low medium high critical do does did any there out so far right now today give me us our""")
    return sorted(content_terms(question.replace("-", " ")) - known)


# ------------------------------------------------------------------ execute + answer
def _match(r: dict, pq: PortfolioQuery) -> bool:
    if pq.vendor_ids and r["vendor_id"] not in pq.vendor_ids:
        return False
    if pq.bands and r["band"] not in pq.bands:
        return False
    if pq.tiers and r["tier"] not in pq.tiers:
        return False
    if pq.review_types and r["review_type"] not in pq.review_types:
        return False
    if pq.recommendations and r["recommendation"] not in pq.recommendations:
        return False
    if pq.groups and r["group"] not in pq.groups:
        return False
    if pq.has_knockout is True and not r["knockouts"]:
        return False
    if pq.has_knockout is False and r["knockouts"]:
        return False
    if pq.business_critical is not None and r.get("business_critical") != pq.business_critical:
        return False
    if pq.knockouts and not set(pq.knockouts) & set(r["knockouts"]):
        return False
    for ef in pq.evidence:
        items = [ef.item] if ef.item else [i for i in r["checklist"] if r["checklist"][i] != "NOT_REQUIRED"]
        wanted = ef.statuses or ["VERIFIED", "EXPIRING_SOON", "EXPIRED", "MISSING", "CONTRADICTORY"]
        if not any(r["checklist"].get(i) in wanted for i in items):
            return False
    if pq.findings and not any(f == c or (c.endswith("-") and f.startswith(c)) for f in r["findings"] for c in pq.findings):
        return False
    if pq.approvers and not all(a in r["approvers"] for a in pq.approvers):
        return False
    return True


def describe(pq: PortfolioQuery) -> str:
    parts = []
    if pq.vendor_ids:
        parts.append("vendors " + ", ".join(pq.vendor_ids))
    if pq.bands:
        parts.append("band " + " or ".join(pq.bands))
    if pq.tiers:
        parts.append("tier " + " or ".join(map(str, pq.tiers)))
    if pq.review_types:
        parts.append("review type " + " or ".join(pq.review_types))
    if pq.recommendations:
        parts.append("recommendation " + " or ".join(pq.recommendations))
    if pq.groups:
        parts.append("comparison group " + " or ".join(pq.groups))
    if pq.has_knockout is not None:
        parts.append("has a knockout" if pq.has_knockout else "no knockout")
    if pq.business_critical is not None:
        parts.append("business-critical" if pq.business_critical else "not business-critical")
    if pq.knockouts:
        parts.append("knockout " + " or ".join(pq.knockouts))
    for ef in pq.evidence:
        parts.append(f"{ef.item or 'any required evidence item'} is " + " or ".join(ef.statuses) if ef.statuses
                     else f"{ef.item} is required (status shown)")
    if pq.findings:
        parts.append("finding " + " or ".join(c + "*" if c.endswith("-") else c for c in pq.findings))
    if pq.approvers:
        parts.append("needs approval from " + " and ".join(pq.approvers))
    return "; ".join(parts) or "all vendors"


def _line(r: dict, pq: PortfolioQuery) -> str:
    s = f"{r['vendor_id']} {r['legal_name']}: {r['band']} (score {r['overall']}), tier {r['tier']}, {r['recommendation']}"
    for ef in pq.evidence:
        items = [ef.item] if ef.item else [i for i, st in r["checklist"].items() if st in ef.statuses]
        s += "; " + ", ".join(f"{i} {r['checklist'].get(i)}" for i in items if i in r["checklist"])
    if r["knockouts"] and (pq.has_knockout or pq.knockouts):
        s += "; knockouts " + ", ".join(r["knockouts"])
    if pq.findings:
        hit = [f for f in r["findings"] if any(f == c or (c.endswith("-") and f.startswith(c)) for c in pq.findings)]
        s += "; findings " + ", ".join(hit)
    return s


def answer(pq: PortfolioQuery, all_rows: list[dict], as_of: str) -> dict:
    """Deterministic answer text, the table, and the citations for a parsed query."""
    hit = [r for r in all_rows if _match(r, pq)]
    filt = describe(pq)
    total = len(all_rows)
    basis = (f"Computed by compliance_checker and financial_risk_calculator for all {total} vendors as of {as_of} "
             f"(filters: {filt}).")
    lines: list[str] = []
    derived: list = [total, len(hit)]                # counts and statistics the text states (for the number self-check)
    if pq.aggregate == "group_by" and pq.group_by:
        key = {"band": "band", "tier": "tier", "recommendation": "recommendation", "group": "group", "review_type": "review_type"}[pq.group_by]
        order = {"band": BANDS, "recommendation": RECS, "tier": [1, 2, 3]}.get(pq.group_by)
        counts: dict = {}
        for r in hit:
            counts[r[key]] = counts.get(r[key], 0) + 1
        keys = [k for k in (order or sorted(counts)) if k in counts]
        text = f"Vendors by {pq.group_by.replace('_', ' ')} ({len(hit)} vendors): " + ", ".join(f"{k}: {counts[k]}" for k in keys) + "."
        derived += list(counts.values())
    elif not hit:
        text = f"No vendor matches ({filt})."
    elif pq.aggregate == "avg":
        vals = [r["overall"] for r in hit]
        text = (f"The average overall risk score of the {len(hit)} vendors matching ({filt}) is {statistics.mean(vals):.1f} "
                f"(lowest {min(vals)}, highest {max(vals)}, median {statistics.median(vals):g}).")
        derived += [f"{statistics.mean(vals):.1f}", f"{statistics.median(vals):g}"]
    elif pq.aggregate in ("max", "min"):
        ranked = sorted(hit, key=lambda r: (r["overall"], r["vendor_id"]), reverse=pq.aggregate == "max")
        n = pq.top_n or min(5, len(ranked))
        top = ranked[0]
        word = "highest" if pq.aggregate == "max" else "lowest"
        tied = [r for r in ranked if r["overall"] == top["overall"]]
        if len(tied) > 1 and n == 1:
            derived.append(len(tied))
            text = (f"{len(tied)} vendors share the {word} overall risk score, {top['overall']}: "
                    + ", ".join(f"{r['vendor_id']} {r['legal_name']}" for r in tied) + ".")
            hit = tied
        else:
            text = (f"{top['vendor_id']} {top['legal_name']} has the {word} overall risk score: {top['overall']} "
                    f"(band {top['band']}, {top['recommendation']}).")
            hit = ranked[:n]
            if n > 1:
                lines = [_line(r, pq) for r in hit]
    else:
        verb = "match" if len(hit) != 1 else "matches"
        if pq.aggregate == "count":
            text = f"{len(hit)} of {total} vendors {verb} ({filt})."
        else:
            text = f"{len(hit)} vendor{'s' if len(hit) != 1 else ''} {verb} ({filt}):"
        lines = [_line(r, pq) for r in hit[:35]]
    notes = []
    if pq.bands and pq.bands != BANDS:
        above = [b for b in BANDS if BANDS.index(b) > max(BANDS.index(x) for x in pq.bands) and b not in pq.bands]
        more = [r for r in all_rows if r["band"] in above and _match(r, pq.model_copy(update={"bands": list(above)}))]
        if more:
            derived.append(len(more))
            notes.append(f"{len(more)} more vendor{'s are' if len(more) != 1 else ' is'} rated higher "
                         f"({', '.join(sorted({r['band'] for r in more}, key=BANDS.index))}) and not included.")
    body = text + ("\n" + "\n".join(f"- {l}" for l in lines) if lines else "")
    body += ("\n" + " ".join(notes) if notes else "") + "\n\n" + basis
    cits = [{"source_id": "POLICY-TPRM-001-v3", "title": "Third-party risk management policy (tiers, evidence, bands)", "kind": "doc"}]
    if pq.approvers:
        cits.append({"source_id": "POLICY-DOA-002", "title": "Delegation of authority policy (approvers)", "kind": "doc"})
    for ef in pq.evidence:
        for r in hit[:4]:
            for i in ([ef.item] if ef.item else [i for i, st in r["checklist"].items() if st in ef.statuses]):
                for d in r["evidence_sources"].get(i, [])[:1]:
                    if all(c["source_id"] != d for c in cits):
                        cits.append({"source_id": d, "title": f"{r['vendor_id']} {i} evidence", "kind": "doc"})
    cits.append({"source_id": "tool:compliance_checker+financial_risk_calculator", "title": f"Rubric engine results as of {as_of}",
                 "kind": "tool"})
    cols = ["vendor_id", "legal_name", "group", "review_type", "tier", "band", "overall", "recommendation", "knockouts"]
    table = [{**{c: (", ".join(r[c]) if isinstance(r[c], list) else r[c]) for c in cols},
              **({"evidence": "; ".join(f"{i}: {r['checklist'].get(i)}" for ef in pq.evidence for i in
                                         ([ef.item] if ef.item else [k for k, s in r['checklist'].items() if s in ef.statuses])
                                         if i in r["checklist"])} if pq.evidence else {}),
              **({"approvers": ", ".join(r["approvers"])} if pq.approvers else {}),
              **({"findings": ", ".join(r["findings"])} if pq.findings else {})} for r in hit]
    return {"answer": body, "table": table, "citations": cits, "matched": [r["vendor_id"] for r in hit], "filters": filt,
            "derived": [str(x) for x in derived]}
