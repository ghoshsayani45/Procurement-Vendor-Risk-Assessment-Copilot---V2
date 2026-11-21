""" the capstone template's four-class failure taxonomy, built on the Week 1 detailed labels.

Each failed query gets ONE primary class (checked in this order) plus its detailed labels:
  tool-call error   the run crashed, a tool failed, JSON had to fall back to a default, or a step/token cap was hit
  retrieval miss    none of the query's gold documents was available to the writer (passages, records, tool sources)
  hallucination     invented citation or finding code, uncited claim, or a critic check that stayed unresolved
                    (numbers not grounded in the facts/passages, citations not retrieved)
  reasoning error   everything else: wrong route (refuse / clarify / answer), a fact missed although its document was
                    retrieved, injection followed, wrong refusal category
Retrieval miss comes before reasoning on purpose: the template calls it the key diagnostic when the generator is weak.
"""
from __future__ import annotations

CLASSES = ("tool-call error", "retrieval miss", "hallucination", "reasoning error")
TOOL_LABELS = {"CALL_ERROR", "JSON_INVALID"}
HALLUCINATION_LABELS = {"INVENTED_CITATION", "INVENTED_FINDING_CODE", "UNCITED"}


def primary_class(q: dict, run: dict, labels: list[str]) -> str:
    if run.get("crashed") or TOOL_LABELS & set(labels):
        return "tool-call error"
    if run.get("tool_failures") or run.get("json_defaulted") or any(
            ("cap" in p or "budget" in p) for p in run.get("partial_reasons", [])):
        return "tool-call error"
    gold = set(q.get("gold_doc_ids") or [])
    if run.get("route") in ("REFUSE", "CLARIFY") and q["expected_behaviour"] == "ANSWER":
        return "reasoning error"                    # wrong route: nothing was retrieved because the agent never planned
    if gold and q["expected_behaviour"] == "ANSWER" and not gold & set(run.get("available_docs") or []):
        return "retrieval miss"
    if HALLUCINATION_LABELS & set(labels) or any(("not found in the facts" in p or "not retrieved" in p)
                                                 for p in run.get("critic_problems_unresolved", [])):
        return "hallucination"
    return "reasoning error"
