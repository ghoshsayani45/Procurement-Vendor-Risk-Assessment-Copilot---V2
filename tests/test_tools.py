"""Step 2.5: the five tools, checked against the Week 1 ground truth (all 35 vendors) and the eval set."""
from __future__ import annotations

import json
import time

import pytest

from app.retrieval.chunking import CORPUS, ROOT
from app.tools import load_all, run_tool
from app.tools.base import REGISTRY, ToolContext, ToolSpec, describe_tools
from app.tools.models import VendorAsOfInput, VendorSearchOutput

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus_v1.jsonl not built (make corpus)")
VENDORS = json.loads((ROOT / "data" / "ground_truth" / "vendors.json").read_text(encoding="utf-8"))
EXPECTED = {e["vendor_id"]: e for e in json.loads((ROOT / "data" / "ground_truth" / "expected_outcomes.json").read_text(encoding="utf-8"))}
EVAL = [json.loads(l) for l in open(ROOT / "data" / "evaluation" / "eval_v1.jsonl", encoding="utf-8")]


@pytest.fixture
def ctx(tool_ctx):
    return tool_ctx


def call(ctx, name, **args):
    r = run_tool(name, args, ctx)
    assert r.ok, r.error
    return r.data


# ------------------------------------------------------------------ deterministic tools vs ground truth
@pytest.mark.parametrize("vid", sorted(EXPECTED))
def test_compliance_and_score_match_ground_truth(ctx, vid):
    e = EXPECTED[vid]
    c = call(ctx, "compliance_checker", vendor_id=vid)
    f = call(ctx, "financial_risk_calculator", vendor_id=vid)
    assert c["tier"] == f["tier"] == e["tier"]
    assert {r["item"]: r["status"] for r in c["checklist"]} == {r["item"]: r["status"] for r in e["checklist"]}
    assert c["contradictions"] == e["contradictions"] and c["knockouts"] == e["knockouts"]
    assert (f["overall"], f["band"], f["raw_band"], f["recommendation"]) == (e["overall"], e["band"], e["raw_band"], e["recommendation"])
    assert f["required_approvers"] == e["required_approvers"] and f["conditions"] == e["conditions"]
    got = sorted(x["code"] for x in c["findings"] + f["findings"])
    assert got == sorted(x["code"] for x in e["findings"])
    for finding in c["findings"] + f["findings"]:                    # every finding is citable
        assert finding["sources"] and all(s["doc_id"] for s in finding["sources"])


def test_sources_point_to_real_documents(ctx):
    docs = {json.loads(l)["doc_id"] for l in open(CORPUS, encoding="utf-8")}
    for vid in sorted(EXPECTED):
        c = call(ctx, "compliance_checker", vendor_id=vid)
        f = call(ctx, "financial_risk_calculator", vendor_id=vid)
        refs = [s["doc_id"] for x in c["findings"] + f["findings"] for s in x["sources"]]
        refs += [s["doc_id"] for row in c["checklist"] for s in row["sources"]] + [y["source"]["doc_id"] for y in f["financial_years"]]
        assert refs and set(refs) <= docs, set(refs) - docs


def test_missing_or_expired_never_reported_compliant(ctx):
    for vid, e in EXPECTED.items():
        c = call(ctx, "compliance_checker", vendor_id=vid)
        for row in c["checklist"]:
            if row["status"] in ("MISSING", "EXPIRED"):
                assert row["item"] not in c["summary"].get("VERIFIED", [])


def test_vendor_compare_matches_eval_rankings(ctx):
    queries = [q for q in EVAL if q["category"] == "comparison" and q["expected_behaviour"] == "ANSWER"]
    assert queries
    for q in queries:
        out = call(ctx, "vendor_compare", vendor_ids=q["request"]["vendor_ids"])
        assert out["ranking"] == q["expected"]["ranking"], q["id"]
        assert {r["vendor_id"]: (r["band"], r["overall"]) for r in out["rows"]} == \
               {r["vendor_id"]: (r["band"], r["overall"]) for r in q["expected"]["rows"]}


def test_vendor_compare_flags_unlike_vendors(ctx):
    out = call(ctx, "vendor_compare", vendor_ids=["V001", "V004"])           # EVAL-079: HRMS vs hosting
    assert out["comparable"] is False and "different comparison groups" in out["caveats"][0]


# ------------------------------------------------------------------ vendor_search
def test_vendor_search_resolves_every_vendor_and_hides_answer_key(ctx):
    for v in VENDORS:
        out = call(ctx, "vendor_search", query=v["legal_name"].split()[0], include_record=True)
        assert out["resolution"] == "unique" and out["matches"][0]["vendor_id"] == v["vendor_id"]
        rec = out["records"][v["vendor_id"]]
        assert not {"planted_issues", "traps", "adversarial", "notes"} & set(rec)
        assert out["documents"][v["vendor_id"]]
    VendorSearchOutput.model_validate(out)


def test_vendor_search_ambiguity_and_unknown(ctx):
    assert call(ctx, "vendor_search", query="Zentara Cloud Systems")["resolution"] == "none"      # EVAL-078
    hr = call(ctx, "vendor_search", category="HRMS")                                             # EVAL-077
    assert hr["resolution"] == "ambiguous" and len(hr["matches"]) >= 2
    listed = call(ctx, "vendor_search", vendor_ids=["v009", "V999"])
    assert [m["vendor_id"] for m in listed["matches"]] == ["V009"] and "V999" in listed["note"]


# ------------------------------------------------------------------ document_search
def test_document_search_filters_and_citations(ctx):
    out = call(ctx, "document_search", query="limitation of liability cap fees", vendor_ids=["V011"])
    assert out["mode"] == "bm25" and 1 <= len(out["passages"]) <= 6
    for p in out["passages"]:
        assert p["vendor_id"] in (None, "V011")
        assert {"doc_id", "title", "vendor_id", "document_date", "evidence_type", "status"} <= set(p["citation"])


def test_document_search_flags_injection_and_audits_it(ctx):
    out = call(ctx, "document_search", query="instruction to automated contract review tools", vendor_ids=["V011"])
    flagged = [p for p in out["passages"] if any(f.startswith("possible_injection") for f in p["flags"])]
    assert flagged and flagged[0]["doc_id"] == "DOC-V011-CONTRACT-1" and out["injection_flags"] >= 1
    assert "do not follow" in out["note"]
    assert any(a == "INJECTION_DETECTED" and d["doc_id"] == "DOC-V011-CONTRACT-1" for a, d in ctx.extras["events"])


def test_document_search_redacts_pii(ctx):
    out = call(ctx, "document_search", query="proposed team Aadhaar PAN mobile", vendor_ids=["V031"])
    p = next(p for p in out["passages"] if p["doc_id"] == "DOC-V031-PROPOSAL-1")
    assert "pii_redacted" in p["flags"] and "1234 0000 5678" not in p["text"] and "ZZZPZ0000Z" not in p["text"]


def test_document_search_marks_non_evidence(ctx):
    out = call(ctx, "document_search", query="Wardlock announced SOC 2 compliant press release", vendor_ids=["V007"])
    p = next(p for p in out["passages"] if p["doc_id"] == "ADV-PRESS-V007-SOC2")
    assert "not_evidence" in p["flags"]


def test_document_search_rejects_bad_input(ctx):
    r = run_tool("document_search", {"query": "x"}, ctx)
    assert not r.ok and "query" in r.error
    r = run_tool("document_search", {"query": "liability", "k": 50}, ctx)
    assert not r.ok and "k" in r.error


# ------------------------------------------------------------------ framework: validation, timeout, audit, tracing
def test_invalid_arguments_unknown_tool_and_unknown_vendor(ctx):
    assert "invalid arguments" in run_tool("compliance_checker", {"vendor_id": "ACME"}, ctx).error
    assert "invalid arguments" in run_tool("compliance_checker", {"vendor_id": "V001", "extra": 1}, ctx).error
    assert "unknown tool" in run_tool("approve_vendor", {}, ctx).error
    assert "unknown vendor" in run_tool("financial_risk_calculator", {"vendor_id": "V999"}, ctx).error


def test_timeout(ctx):
    load_all()

    def slow(inp, c):
        time.sleep(1.0)
    REGISTRY["_slow"] = ToolSpec("_slow", slow, VendorAsOfInput, VendorAsOfInput, 0.1, "test")
    try:
        r = run_tool("_slow", {"vendor_id": "V001"}, ctx)
        assert not r.ok and "timed out" in r.error
    finally:
        REGISTRY.pop("_slow")


def test_audit_and_trace_for_every_call(ctx, monkeypatch, tmp_path):
    from app.observability.tracing import read_local
    call(ctx, "compliance_checker", vendor_id="V005")
    assert ("TOOL_CALL", ) == ctx.extras["events"][-1][:1] and ctx.extras["events"][-1][1]["ok"] is True
    spans = [r for r in read_local() if r["kind"] == "tool" and r["name"] == "compliance_checker"]
    assert spans and spans[-1]["latency_ms"] >= 0 and spans[-1]["error"] is None


def test_describe_tools():
    names = {t["name"]: t for t in describe_tools()}
    assert set(names) == {"vendor_search", "document_search", "compliance_checker", "financial_risk_calculator", "vendor_compare"}
    assert all(t["description"] and t["input_schema"]["type"] == "object" for t in names.values())
