"""Query expansion for the external sources in the corpus (Week 3 fix for retrieval misses).

Questions name regulations by acronym ("GFR", "DPDP Act", "CCM") while the chunks carry the full titles ("General
Financial Rules 2017", "Digital Personal Data Protection Act 2023"). BM25 cannot connect the two, so every
document_search query that uses an acronym gets the full source name added.

The table is built from the corpus document titles (data/processed/corpus_v1.jsonl, POL-* and GUI-* documents),
not from eval questions or answers. Disclosed in docs/agent_eval.md.
"""
from __future__ import annotations

import re

_I = re.IGNORECASE
ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bGFR\b", _I), "General Financial Rules 2017"),
    (re.compile(r"\bDPDP\s+Rules?\b", _I), "Digital Personal Data Protection Rules 2025"),
    (re.compile(r"\bDPDP(\s+Act)?\b", _I), "Digital Personal Data Protection Act 2023"),
    (re.compile(r"\bGoods\s+Manual\b|\bMPG\b", _I), "Manual for Procurement of Goods"),
    (re.compile(r"\bNon-?Consultancy\s+(Services\s+)?Manual\b", _I), "Manual for Procurement of Non-Consultancy Services"),
    (re.compile(r"\bNIST\b|\bSP\s*800-161", _I), "NIST SP 800-161 Cybersecurity Supply Chain Risk Management Practices"),
    (re.compile(r"\bCCM\b|\bCSA\b", _I), "Cloud Controls Matrix"),
    (re.compile(r"\bC-?SCRM\b", _I), "cybersecurity supply chain risk management"),
    (re.compile(r"\bSAF\b", _I), "Supplier Assurance Framework"),
    (re.compile(r"\bIPF\b", _I), "World Bank Investment Project Financing"),
]
# words in the corpus document titles that identify each source (the titles abbreviate some: "CCM v4.1", "NIST SP 800-161r1")
TITLE_KEYS = {"General Financial Rules 2017": "general financial rules",
              "Digital Personal Data Protection Rules 2025": "digital personal data protection rules",
              "Digital Personal Data Protection Act 2023": "digital personal data protection act",
              "Manual for Procurement of Goods": "manual for procurement of goods",
              "Manual for Procurement of Non-Consultancy Services": "non-consultancy services",
              "NIST SP 800-161 Cybersecurity Supply Chain Risk Management Practices": "nist sp 800-161",
              "Cloud Controls Matrix": "ccm",
              "cybersecurity supply chain risk management": "nist sp 800-161",
              "Supplier Assurance Framework": "supplier assurance framework",
              "World Bank Investment Project Financing": "world bank"}


def expand_query(q: str) -> str:
    """Add the full source name for each acronym the query uses (once; nothing if the name is already there)."""
    extra = [full for pat, full in ALIASES if pat.search(q) and full.lower() not in q.lower()]
    return (" ".join(dict.fromkeys(extra)) + " " + q).strip()[:500] if extra else q


EXTERNAL_DOC_TYPES = ["procurement_policy", "guidance"]     # GFR, procurement manuals, DPDP, World Bank, SAF, NIST, CSA


def named_sources(q: str) -> list[str]:
    """Full names of the external sources a question names (by acronym or by name)."""
    out = [full for pat, full in ALIASES if pat.search(q) or full.lower() in q.lower()]
    return list(dict.fromkeys(out))


def expand_for(query: str, question: str) -> str:
    """Expand a search query with the sources named in the query OR in the question it serves: the planner's keyword
    queries usually drop the acronym ("single source procurement" for "When does GFR allow ...")."""
    extra = [n for n in named_sources(query) + named_sources(question) if n.lower() not in query.lower()]
    return (" ".join(dict.fromkeys(extra)) + " " + query).strip()[:500] if extra else query


def order_by_named_source(passages: list[dict], question: str) -> tuple[list[dict], list[str]]:
    """Passages from the source(s) the question names first (in the order named; within one source, current versions
    before superseded ones), the rest after in their fused order. Returns (passages, the named sources' full names)."""
    named = named_sources(question)
    keys = [TITLE_KEYS.get(n, n.lower()) for n in named]
    if not keys:
        return list(passages), []

    def rank(p):
        t = (p.get("title") or "").lower()
        hit = next((i for i, k in enumerate(keys) if re.search(rf"\b{re.escape(k)}", t)), len(keys))
        return (hit, p.get("status") == "SUPERSEDED")
    return sorted(passages, key=rank), named        # sorted() is stable: the fused order holds within each rank
