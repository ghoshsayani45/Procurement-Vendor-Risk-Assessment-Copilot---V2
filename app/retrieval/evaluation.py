"""Retrieval metrics, measured separately from answer quality.

A search function takes an eval query and returns ranked Passages. Metrics per query:
  hit@k        any gold document among the top-k passages
  recall@k     share of the query's gold documents among the top-k passages
  mrr          1 / rank of the first passage from a gold document (0 if none in the list)
  quote@k      share of the query's gold quotes found verbatim (whitespace/case-insensitive) inside a top-k passage:
               the passage-level check. It shows whether the chunk the generator would read actually holds the evidence,
               which document-level metrics cannot (a hit on the right document with the wrong chunk still scores 1 there).
  ctx_words@6  words in the top 6 passages: what the generator would have to read (cost)
Queries are grouped by category and by the kind of document their gold evidence is in (doc group).
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict

from app.retrieval.chunking import ROOT
from app.retrieval.retrievers import Filters

EVAL = ROOT / "data" / "evaluation" / "eval_v1.jsonl"
KS = (5, 10, 20)
GROUPS = {
    "contracts": {"contract", "dpa", "contract_amendment", "contract_public", "nda_public", "contract_template"},
    "vendor_evidence": {"soc2_report", "iso27001_certificate", "caiq_response", "pentest_report", "insurance_certificate",
                        "bcp_dr_report", "financial_statements", "qbr_sla_report", "incident_log", "purchase_order",
                        "invoice", "proposal", "email", "press_release", "vendor_webpage", "chat_transcript", "note"},
    "internal_policy": {"internal_policy", "policy_draft"},
    "external_policy": {"procurement_policy", "guidance"},
}


def doc_group(doc_type: str) -> str:
    return next((g for g, types in GROUPS.items() if doc_type in types), "other")


def load_queries(splits=("dev",)) -> list[dict]:
    qs = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    return [q for q in qs if q["split"] in splits and q["gold_doc_ids"]]


def query_group(q: dict, doc_types: dict[str, str]) -> str:
    groups = {doc_group(doc_types.get(d, "")) for d in q["gold_doc_ids"]}
    return groups.pop() if len(groups) == 1 else "mixed"


def query_filters(q: dict, use_vendor_filter: bool) -> Filters:
    """Filters the copilot would apply: the vendors named on the case (if any). Access: public + internal."""
    vids = (q.get("request") or {}).get("vendor_ids") or None
    return Filters(vendor_ids=tuple(vids) if (use_vendor_filter and vids) else None)


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.replace("*", "").replace(" ", " ")).strip().lower()


def score_query(q: dict, passages, k_max: int = 50) -> dict:
    gold = set(q["gold_doc_ids"])
    docs = [p.doc_id for p in passages[:k_max]]
    row = {"id": q["id"], "category": q["category"], "split": q["split"], "n_gold": len(gold)}
    for k in KS:
        top = set(docs[:k])
        row[f"hit@{k}"] = int(bool(top & gold))
        row[f"recall@{k}"] = round(len(top & gold) / len(gold), 3)
    rank = next((i + 1 for i, d in enumerate(docs) if d in gold), None)
    row["mrr"] = round(1 / rank, 3) if rank else 0.0
    quotes = q.get("gold_quotes") or []
    if quotes:
        texts = [_norm(p.text) for p in passages[:max(KS)]]
        for k in (5, 10):
            found = sum(1 for g in quotes if any(_norm(g["quote"]) in t for t in texts[:k]))
            row[f"quote@{k}"] = round(found / len(quotes), 3)
    row["ctx_words@6"] = sum(len(p.text.split()) for p in passages[:6])
    row["top3"] = docs[:3]
    return row


def evaluate(search, queries: list[dict], doc_types: dict[str, str]) -> list[dict]:
    rows = []
    for q in queries:
        t0 = time.time()
        passages = search(q)
        row = score_query(q, passages)
        row["latency_ms"] = round((time.time() - t0) * 1000, 1)
        row["group"] = query_group(q, doc_types)
        rows.append(row)
    return rows


METRICS = [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr", "quote@5", "quote@10", "ctx_words@6", "latency_ms"]


def summarise(rows: list[dict], by: str | None = None) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups["ALL"].append(r)
        if by:
            groups[r[by]].append(r)
    out = {}
    for g, rs in groups.items():
        m = {"n": len(rs)}
        for key in METRICS:
            vals = [r[key] for r in rs if key in r]
            if vals:
                m[key] = round(sum(vals) / len(vals), 3) if key not in ("ctx_words@6", "latency_ms") else round(sorted(vals)[len(vals) // 2], 1)
                if key.startswith("quote"):
                    m["n_quote_queries"] = len(vals)
        out[g] = m
    return out
