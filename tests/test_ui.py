"""the Streamlit UI renders a real agent review (mock JSON from the API) and the Markdown export."""
import json

import pytest

from app.retrieval.chunking import ROOT

st_testing = pytest.importorskip("streamlit.testing.v1")
MOCK = ROOT / "app" / "ui" / "mock_review.json"
needs_mock = pytest.mark.skipif(not MOCK.exists(), reason="run python -m scripts.make_ui_mock")


def pdf_works() -> bool:
    try:
        from app.ui.report_pdf import markdown_pdf
        markdown_pdf("# t")
        return True
    except Exception:
        return False


def app_test(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)                 # mock mode is the default without an API key
    return st_testing.AppTest.from_file(str(ROOT / "app" / "ui" / "streamlit_app.py"), default_timeout=60).run()


@needs_mock
def test_ui_renders_the_mock_review_with_all_tabs(monkeypatch):
    at = app_test(monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Case", "Plan", "Evidence", "Scorecard", "Approval", "Report"]
    assert any(m.label == "Risk score" for m in at.metric) and any(m.label == "Steps (cap 10)" for m in at.metric)
    md = " ".join(m.value for m in at.markdown)
    assert "validate" in md and "finalize" in md                              # plan view: the node path
    assert "Expired" in md or "Missing" in md                                 # checklist grouped by status
    assert any("instruction-like text" in e.value for e in at.error)          # injected passage is visibly flagged
    assert len(at.get("download_button")) == (2 if pdf_works() else 1)       # report export: Markdown (+ PDF where it works)


def test_ui_without_data_asks_for_a_source(monkeypatch, tmp_path):
    monkeypatch.delenv("API_KEY", raising=False)
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "ui" / "streamlit_app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Live API").run()
    assert not at.exception and any("Enter an API key" in i.value for i in at.info)


@needs_mock
def test_markdown_report_has_the_approver_sections():
    from app.ui.report_md import review_markdown
    review = json.loads(MOCK.read_text(encoding="utf-8"))
    md = review_markdown(review)
    for heading in ("# Vendor risk review", "## Risk score", "| Dimension | Weight |", "## Evidence checklist", "## Findings",
                    "## Decision", "## Audit trail", "## Follow-up questions"):
        assert heading in md, heading
    assert "never approves" in md and "No decision recorded" in md


def test_checklist_groups_follow_the_reviewer_order():
    from app.ui.report_md import group_checklist
    rows = [{"item": i, "status": s} for i, s in [("A", "VERIFIED"), ("B", "MISSING"), ("C", "EXPIRED"), ("D", "CONTRADICTORY"),
                                                  ("E", "WEIRD")]]
    assert [g for g, _ in group_checklist(rows)] == ["Verified", "Expired", "Missing", "Contradictory or insufficient", "Other"]


def test_markdown_for_refusal_partial_and_override():
    from app.ui.report_md import review_markdown
    base = {"id": "r1", "request_kind": "QUESTION", "status": "REFUSED", "audit": [], "turns": [
        {"role": "assistant", "content": {"kind": "answer", "output": {"behaviour": "REFUSE", "category": "APPROVAL_OR_COMMITMENT",
                                                                      "message": "I cannot approve vendors."}}}]}
    assert "## Request refused" in review_markdown(base) and "APPROVAL_OR_COMMITMENT" in review_markdown(base)
    part = {**base, "status": "PARTIAL", "turns": [{"role": "assistant", "content": {
        "kind": "answer", "partial_reasons": ["compliance_checker: injected fault: timeout"],
        "output": {"behaviour": "PARTIAL", "answer": "x", "citations": []}}}]}
    assert "## Partial result" in review_markdown(part) and "injected fault" in review_markdown(part)
    risk = {"band": "HIGH", "overall": 34.5, "raw_band": "MEDIUM", "dimensions": [], "recommendation": "R", "required_approvers": ["CISO"]}
    ov = {**base, "status": "SCORED", "turns": [], "score_json": {"risk": risk, "overrides": [
        {"band": "CRITICAL", "computed_band": "HIGH", "reason": "Pending breach investigation", "reviewer_name": "Ravi", "at": "t"}]}}
    assert "set the band to **CRITICAL** (computed: HIGH)" in review_markdown(ov)


@needs_mock
def test_pdf_export_renders_the_review_and_never_fails_on_unicode(monkeypatch):
    pytest.importorskip("fpdf")
    from app.ui import report_pdf
    review = json.loads(MOCK.read_text(encoding="utf-8"))
    data = report_pdf.review_pdf(review)
    assert data[:5] == b"%PDF-" and len(data) > 5000
    monkeypatch.setattr(report_pdf, "_font_files", lambda: None)            # no Unicode font on this machine
    data = report_pdf.markdown_pdf("# Band → HIGH ≥ 25 · ₹\n\n| a | b |\n|---|---|\n| x → y | **z** |\n\n- item")
    assert data[:5] == b"%PDF-"
