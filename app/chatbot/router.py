"""Rule-first routing for the chat assistant (D45), in the same spirit as the agent's D36: decided in code where a
rule can decide, the LLM only as a fallback for unclear risky requests.

Order: refusal rules -> help -> vendor ids / names in the question (1 vendor: VENDOR; 2-3 with a compare word:
COMPARISON; 2-3 without: PORTFOLIO over those vendors) -> portfolio phrasing -> follow-up on the vendor in focus
("what about its insurance?") -> an unknown name asked about like a vendor (CLARIFY) -> system / policy vocabulary
(SYSTEM = documentation first, POLICY = internal rules and regulations first; both sources are searched either way).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from app.agents import rules
from app.chatbot import portfolio

_I = re.I
HELP = re.compile(r"^\s*(hi|hello|hey|help|thanks|thank you|good (morning|afternoon|evening)|start)\b[\s!.?]*$|"
                  r"\bwhat (can|do|could) you (do|help)|\bhow (do|can|should) i use (you|this)\b|\bwho are you\b|"
                  r"\bwhat are you\b|\bwhat can i ask\b|\bwhat (kind|sort) of questions\b", _I)
COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|side[\s-]by[\s-]side|riskier|safer|rank(ing)?|better|worse|"
                     r"difference between)\b", _I)
FOLLOWUP = re.compile(r"\b(it|its|it's|they|their|them|this vendor|that vendor|the vendor|this one|that one|same vendor|"
                      r"this supplier|the supplier)\b|^\s*(and|also|what about|how about)\b", _I)
SYSTEM = re.compile(
    r"\bcopilot\b|\bagent'?s?\b|\bpipeline\b|\barchitecture\b|\bretriev\w*|\bchunk\w*|\brerank\w*|\bembedding\w*|\bbm25\b|"
    r"\bdense\b|\bhybrid\b|\brrf\b|\blanggraph\b|\blangfuse\b|\btrac(e|es|ing)\b|\bbenchmark\w*|\beval(uation)?s?\b|"
    r"\beval_v\d\b|\baccuracy\b|\bprecision\b|\brecall\b|\bablation\w*|\bbaselines?\b|\bfailure (log|taxonomy|class\w*|notes?)\b|"
    r"\bdecisions?\b|\bD\d{1,2}\b|\bweek [1-4]\b|\bmodel\b|\bllm\b|\bqwen\w*|\btokens?\b|\bcontext window\b|\bstep cap\b|"
    r"\bsteps?\b|\bbudgets?\b|\bcritic\b|\bplanner\b|\brouter\b|\bjson\b|\bschemas?\b|\bdocker\w*|\bcontainers?\b|\bapi\b|"
    r"\bendpoint\b|\bstreamlit\b|\bui\b|\bfront[\s-]?end\b|\btests?\b|\btest suite\b|\bpytest\b|\bcorpus\b|\bdataset\b|"
    r"\bground truth\b|\bsynthetic\b|\bmulti[\s-]agent\b|\bsingle agent\b|\bprompt(ing)?\b|\bfew[\s-]shot\b|"
    r"\bchain[\s-]of[\s-]thought\b|\bself[\s-]consistency\b|\bsmall[\s-]model\b|\bmitigation\w*|\bcach(e|ing)\b|"
    r"\brate limit\w*|\bthreat model\b|\binjection\b|\bsafety (report|suite|tests?|probes?)\b|\bteam\b|\bmembers?\b|"
    r"\bowners?\b|\bcontribut\w*|\brepo(sitory)?\b|\bmakefile\b|\bpython\b|\balembic\b|\bpostgres\w*|\bsqlite\b|\bpgvector\b|"
    r"\bmigrations?\b|\blatency\b|\bcost\b|\bquota\b|\bgemini\b|\bcross[\s-]encoder\b|\btools?\b|\brefusal categor\w*|"
    r"\bhow does (it|the copilot|the system|the agent)\b|\bbuilt\b|\bdesigned\b|\bimplemented\b|\bgate\b|\bgpu\b|"
    r"\bkubernetes\b|\bchecklist accuracy\b|\bcitation\w*|\bhallucinat\w*|\bguardrails?\b|\bobservability\b", _I)
POLICY = re.compile(
    r"\bpolic(y|ies)\b|\bstandards?\b|\bregulations?\b|\bgfr\b|\bgeneral financial rules\b|\bdpdp\b|\bdata protection act\b|"
    r"\bnist\b|\bccm\b|\bcsa\b|\bcloud controls\b|\bsupplier assurance\b|\bworld bank\b|\bprocurement manual\b|\blaw\b|"
    r"\bact\b|\bclauses?\b|\btier [123]\b|\brequired evidence\b|\bevidence (is |are )?required\b|\bminimum\b|\bmaximum\b|"
    r"\bbreach\w*|\bnotif\w*|\bliabilit\w*|\bcap\b|\bauto[\s-]?renew\w*|\bnotice period\b|\bterminat\w*|\baudit rights?\b|"
    r"\bprice escalation\b|\bsub[\s-]?processors?\b|\boffshore\b|\bcross[\s-]border\b|\bapproval matrix\b|\bdelegation\b|"
    r"\bwho (approves|must approve|has to approve|needs to approve|can approve|signs)\b|\bapprove a contract\b|"
    r"\binsurance cover\b|\bdebarment list\b|\bconflict[\s-]of[\s-]interest register\b|\bsingle[\s-]source\b|\btender\w*|"
    r"\bbids?\b|\bprocure\w*|\bcontract standards?\b|\btprm\b|\bvalidity\b|\bhow (long|old)\b|\bband thresholds?\b|"
    r"\bweights?\b|\bknockout rules?\b|\brubric\b|\bthresholds?\b|\bcrore\b|\blakh\b|\bhours?\b|\bdays?\b", _I)
OVERVIEW = re.compile(r"\b(overview|summary|summari[sz]e|tell me about|profile|status of|how risky|assessment of|"
                      r"risk (profile|summary)|snapshot|who is)\b|"
                      r"\bwhy\b[^?]{0,60}\b(rated|band(ed)?|scored|risk(y)?|score|high|low|medium|critical|escalat\w*|on hold)\b", _I)
VENDOR_ATTR = re.compile(r"\b(risk|band|score|tier|soc\s*2|iso|insurance|approv\w*|contract|dpa|pen(etration)?[\s-]?test|"
                         r"evidence|recommend\w*|status|rated|risky|debarred|knockout|caiq|financ\w*|sla|onboard\w*|renewal)\b", _I)
NOT_NAMES = {"SOC", "ISO", "GFR", "DPDP", "NIST", "CAIQ", "DPA", "CFO", "DPO", "LLM", "API", "BM25", "CSA", "CCM", "SAF", "TPRM",
             "INR", "Qwen", "Qwen3", "Langfuse", "LangGraph", "Gemini", "Docker", "Streamlit", "Postgres", "Indrapath",
             "Technologies", "Week", "Rule", "Rules", "Act", "Policy", "Standard", "Head", "Procurement", "Finance", "Controller",
             "Chief", "Financial", "Officer", "Committee", "Legal", "Counsel", "Data", "Protection", "Ethics", "Compliance",
             "Category", "Manager", "Third-Party", "Risk", "Tier", "HIGH", "LOW", "MEDIUM", "CRITICAL", "Copilot", "Ask",
             "Tell", "Give", "List", "Show", "Why", "Who", "Where", "Should", "Could", "Would", "Will", "I", "And", "Also"}


@dataclass
class Decision:
    route: str                        # HELP REFUSE VENDOR COMPARISON PORTFOLIO SYSTEM POLICY CLARIFY
    vendor_ids: list[str] = field(default_factory=list)
    refusal_category: str | None = None
    clarify: dict | None = None
    reason: str = ""
    by: str = "rule"                  # rule | session | llm | tool
    note: str | None = None           # e.g. the approval part of a mixed request was declined
    source_weights: tuple[float, float] = (1.0, 1.0)   # (documentation, rules) for SYSTEM / POLICY retrieval


# ------------------------------------------------------------------ vendor directory (brand words from the records)
_lock = threading.Lock()
_DIR: dict[int, dict] = {}


def directory(store) -> dict[str, dict]:
    """vendor_id -> {legal_name, brand, group}; the brand is the first distinctive word of the legal name ('Kestrion')."""
    key = id(store)
    with _lock:
        if key in _DIR:
            return _DIR[key]
    from app.tools.vendor_tools import GENERIC
    recs = store.get_vendor_records()
    out = {}
    for vid, r in recs.items():
        words = [w for w in re.findall(r"[A-Za-z0-9]+", r["legal_name"]) if w.lower() not in GENERIC and len(w) >= 4]
        out[vid] = {"legal_name": r["legal_name"], "brand": (words[0].lower() if words else ""), "group": r["comparison_group"],
                    "category": r["category"]}
    brands = [v["brand"] for v in out.values()]
    for v in out.values():                                 # a brand shared by two vendors is not distinctive
        if brands.count(v["brand"]) > 1:
            v["brand"] = ""
    with _lock:
        _DIR[key] = out
    return out


def reset():
    with _lock:
        _DIR.clear()


def find_vendors(q: str, dir_: dict[str, dict]) -> list[str]:
    """Vendor ids named in the question, in the order they appear (by id, brand word or full legal name)."""
    hits: list[tuple[int, str]] = []
    for m in re.finditer(r"\bV(\d{3})\b", q, _I):
        vid = f"V{m.group(1)}"
        if vid in dir_:
            hits.append((m.start(), vid))
    low = q.lower()
    for vid, v in dir_.items():
        for needle in filter(None, (v["legal_name"].lower(), v["brand"])):
            m = re.search(rf"\b{re.escape(needle)}\b", low)
            if m:
                hits.append((m.start(), vid))
                break
    out: list[str] = []
    for _, vid in sorted(hits):
        if vid not in out:
            out.append(vid)
    return out


def unknown_names(q: str, dir_: dict[str, dict]) -> list[str]:
    names = []
    for n in rules.candidate_names(q):
        words = [w for w in n.split() if w not in NOT_NAMES]
        if words and not find_vendors(" ".join(words), dir_):
            names.append(" ".join(words))
    return names


def decide(question: str, focus: list[str], store, llm_router=None) -> Decision:
    q = (question or "").strip()
    if not q:
        return Decision("HELP", reason="empty message")
    cat = rules.refusal(q)
    if cat:
        return Decision("REFUSE", refusal_category=cat, reason="refusal rule")
    note = None
    if rules.mixed_approval(q):
        note = ("Approval declined: the copilot cannot approve, sign or onboard vendors; the decision belongs to the "
                "required approvers (POLICY-DOA-002).")
    if HELP.search(q) and not find_vendors(q, directory(store)):
        return Decision("HELP", reason="help phrasing")
    if llm_router is not None and rules.risky(q):
        cat = llm_router(q)
        if cat:
            return Decision("REFUSE", refusal_category=cat, reason="LLM router (risky wording, no rule matched)", by="llm")
    dir_ = directory(store)
    vids = find_vendors(q, dir_)
    if len(vids) >= 2:
        vids = vids[:3]
        if COMPARE.search(q):
            return Decision("COMPARISON", vendor_ids=vids, reason="two or more vendors and a compare word", note=note)
        return Decision("PORTFOLIO", vendor_ids=vids, reason="several vendors named", note=note)
    if len(vids) == 1:
        if COMPARE.search(q) and re.search(r"\b(peers?|others?|competitors?|alternatives?|same group)\b", q, _I):
            peers = [v for v, d in dir_.items() if d["group"] == dir_[vids[0]]["group"] and v != vids[0]]
            if peers:
                return Decision("COMPARISON", vendor_ids=(vids + peers)[:3], reason="vendor compared with its peers", note=note)
        return Decision("VENDOR", vendor_ids=vids, reason="vendor named", note=note)
    if portfolio.looks_like_portfolio(q):
        return Decision("PORTFOLIO", reason="portfolio phrasing", note=note)
    sys_hits, pol_hits = len(SYSTEM.findall(q)), len(POLICY.findall(q))
    if focus and FOLLOWUP.search(q) and not sys_hits:
        if len(focus) >= 2 and COMPARE.search(q):
            return Decision("COMPARISON", vendor_ids=focus[:3], reason="follow-up on the vendors in focus", by="session", note=note)
        return Decision("VENDOR", vendor_ids=focus[:1], reason="follow-up on the vendor in focus", by="session", note=note)
    names = unknown_names(q, dir_)
    if names and VENDOR_ATTR.search(q) and not sys_hits and not pol_hits:
        return Decision("CLARIFY", clarify={"question": f"I can't find a registered vendor named '{names[0]}'. Which vendor do "
                                                        "you mean? It may not be onboarded yet.", "options": []},
                        reason="unknown vendor name")
    if sys_hits and sys_hits >= pol_hits:
        return Decision("SYSTEM", reason=f"system vocabulary ({sys_hits} vs {pol_hits} policy terms)", source_weights=(1.0, 0.6), note=note)
    if pol_hits:
        return Decision("POLICY", reason=f"policy vocabulary ({pol_hits} vs {sys_hits} system terms)", source_weights=(0.6, 1.0), note=note)
    return Decision("SYSTEM", reason="no rule matched: documentation and policies searched equally", source_weights=(1.0, 1.0), note=note)
