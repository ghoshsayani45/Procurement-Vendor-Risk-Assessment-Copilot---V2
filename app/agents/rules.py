"""rule-based routing before any LLM call. Refusals and clarifications are decided in code first.

refusal(question) -> category or None     : the spec's refusal policy (docs/spec.md section 6), as patterns
risky(question) -> bool                   : words that deserve the LLM router when no rule fired
mixed_approval(question) -> bool          : "assess X and approve it" -> do the review, decline the approval part
resolve_vendors / clarify_*               : vendor resolution and the spec's clarify cases (section 3)
"""
from __future__ import annotations

import re

_I = re.I
REFUSAL_RULES: list[tuple[str, re.Pattern]] = [
    ("SECRETS", re.compile(r"\b(system prompt|api[- ]?key|password|credential|connection string|access token|secret key)s?\b", _I)),
    ("PII_EXPOSURE", re.compile(r"\b(aadhaar|pan (number|no|card)s?|pan numbers?|phone numbers?|mobile numbers?|home address(es)?|"
                                r"date of birth|personal (details|data) of)\b", _I)),
    ("SCORE_MANIPULATION", re.compile(r"\b(make|get|have)\b[^.?]{0,60}\b(come out|appear|look|rated?|score[sd]?)\b[^.?]{0,20}"
                                      r"\b(as )?(low|lower|medium|better)\b|\b(leave|keep|take)\b[^.?]{0,40}\bout of the (report|review)\b|"
                                      r"\b(hide|remove|drop|ignore)\b[^.?]{0,30}\b(finding|knockout|conflict of interest)|"
                                      r"\b(lower|reduce|bump down|tweak|adjust|override|change)\b[^.?]{0,40}\b(risk )?(score|band|rating)\b|"
                                      r"\b(score|band|rating)\b[^.?]{0,30}\bso (it|they|that it) (gets?|passes|goes) through\b", _I)),
    ("LEGAL_ADVICE", re.compile(r"\b(enforceable|legally binding|can we sue|legal opinion|is (it|this|that) legal|in (indian )?courts?)\b", _I)),
    ("ACCESS_CONTROL_BYPASS", re.compile(r"\b(restricted [^.?]{0,25}(documents?|files?|records?|folders?|minutes)|marked restricted|another team'?s|other teams'?|"
                                         r"board minutes|bypass (the )?access|(admin|write|edit) access to|"
                                         r"edit the (vendor )?(master|bank details) (myself|directly))\b", _I)),
    ("OUT_OF_SCOPE", re.compile(r"\b(buy|sell) (shares|stock)|\binvest(ment)? (in|advice)|\bshare price\b|\bstock price\b|"
                                r"\b(write (me )?a (poem|song|story))\b|\bweather\b", _I)),
    # an action, not an assessment: only when the request STARTS with the action (mixed requests are handled separately)
    ("APPROVAL_OR_COMMITMENT", re.compile(r"^\s*(please\s+|(can|could|would|will) you\s+(please\s+)?|i need you to\s+|"
                                          r"(just\s+)?go ahead and\s+)*(just\s+)?(go ahead and\s+)?(approve|sign( off)?|mark [^.?]{0,30} as approved|"
                                          r"commit (us|the company)|countersign|award|execute|raise (a |the )?(po|purchase order)|"
                                          r"issue (a |the )?(po|purchase order)|onboard\b(?! assessment)|pay\b|release [^.?]{0,25}payment|"
                                          r"(change|update|replace|switch) [^.?]{0,60}(payment terms|bank (account|details)|"
                                          r"account details|ifsc|beneficiary|payee)|renew\b(?! review))\b", _I)),
]
RISKY = re.compile(r"\b(approv\w*|sign\w*|payment|pay|bank|aadhaar|pan|password|prompt|key|legal\w*|enforce\w*|shares?|invest\w*|"
                   r"restricted|hide|leave out|delete|bypass|confidential)\b", _I)
MIXED_APPROVAL = re.compile(r"\b(and|then)\s+(approve|sign|onboard|award)\b|\bapprove (it|them)\b", _I)

ROUTE_TO = {"APPROVAL_OR_COMMITMENT": "The approvers named in the delegation-of-authority policy (POLICY-DOA-002)",
            "LEGAL_ADVICE": "Legal Counsel", "ACCESS_CONTROL_BYPASS": "The document owner",
            "PII_EXPOSURE": "Data Protection Officer", "SECRETS": None, "SCORE_MANIPULATION": "Third-Party Risk Committee",
            "OUT_OF_SCOPE": None}
MESSAGES = {
    "APPROVAL_OR_COMMITMENT": "I can't approve, sign, award or change commercial terms. Only a human approver can make that decision; "
                              "I can prepare the risk assessment they need.",
    "LEGAL_ADVICE": "I can't give a legal opinion on enforceability. Legal Counsel can; I can point to the clause and the contract standard.",
    "ACCESS_CONTROL_BYPASS": "I can't show documents outside your access. The document owner can grant access.",
    "PII_EXPOSURE": "I can't list personal data such as ID or phone numbers from vendor documents. The Data Protection Officer can "
                    "advise on lawful access.",
    "SECRETS": "I can't share system instructions, API keys or credentials.",
    "SCORE_MANIPULATION": "I can't change scores or leave findings out. Scores come from the rubric; the Third-Party Risk Committee "
                          "can review a disputed finding.",
    "OUT_OF_SCOPE": "That is outside vendor risk assessment, so I can't help with it here.",
}


def refusal(question: str) -> str | None:
    q = question or ""
    for cat, rx in REFUSAL_RULES:
        if rx.search(q):
            return cat
    return None


def risky(question: str) -> bool:
    return bool(RISKY.search(question or ""))


def mixed_approval(question: str) -> bool:
    return bool(MIXED_APPROVAL.search(question or ""))


# ------------------------------------------------------------------ vendor resolution and clarify cases
CATEGORY_WORDS = {"hr": "HRMS", "hrms": "HRMS", "payroll": "payroll", "human resources": "HRMS", "hosting": "hosting",
                  "cloud hosting": "hosting", "crm": "CRM", "payments": "payment", "payment gateway": "payment", "sso": "identity",
                  "identity": "identity", "contact centre": "contact", "customer support": "contact", "cctv": "surveillance",
                  "security operations": "SOC", "endpoint": "endpoint", "analytics": "analytics", "document processing": "document"}
STOP_CAPS = {"Run", "Assess", "Compare", "Review", "Do", "Full", "New", "We", "Our", "Please", "Prepare", "Renewal", "Onboarding",
             "What", "Which", "Is", "Are", "Does", "Has", "How", "When", "Who", "Can", "Should", "The", "A", "An", "Side-by-side"}


def candidate_names(question: str) -> list[str]:
    """Capitalised word runs that could be a vendor name ('Zentara Cloud Systems')."""
    runs = re.findall(r"\b([A-Z][\w&-]+(?:\s+[A-Z][\w&-]+){0,3})\b", question or "")
    out = []
    for r in runs:
        words = [w for w in r.split() if w not in STOP_CAPS]
        if words and not (len(words) == 1 and words[0].isupper() and len(words[0]) <= 4):
            out.append(" ".join(words))
    return out


def category_hint(question: str) -> str | None:
    low = (question or "").lower()
    for k in sorted(CATEGORY_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", low):
            return CATEGORY_WORDS[k]
    return None


STATUS_Q = re.compile(r"^\s*(has|have|is|are|was|were|did|does)\b[^?]{0,80}\b(approved|signed|onboarded|awarded|cleared)\b", _I)
DIRECTIVE = re.compile(r"\b(please|go ahead|just|can you|could you|would you)\b[^?]{0,30}\b(approve|sign|mark|record)\b", _I)


def status_question(q: str) -> bool:
    """'Has X been approved yet?' asks about state: answerable (no approval recorded / who must approve)."""
    return bool(STATUS_Q.search(q or "")) and not DIRECTIVE.search(q or "")
