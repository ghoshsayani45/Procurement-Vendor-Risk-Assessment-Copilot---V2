"""D45: the Streamlit 'Ask the Copilot' view, with the API faked at the HTTP layer (no backend needed)."""
from __future__ import annotations

import pytest

from app.retrieval.chunking import ROOT

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(ROOT / "app" / "ui" / "streamlit_app.py")

TURN = {"kind": "chat", "route": "PORTFOLIO", "behaviour": "ANSWER", "session_id": "s1",
        "answer": "5 of 35 vendors match (band CRITICAL).",
        "table": [{"vendor_id": "V007", "legal_name": "Wardlock Security Operations LLC", "band": "CRITICAL", "overall": 51.9}],
        "citations": [{"source_id": "POLICY-TPRM-001-v3", "title": "Third-party risk management policy", "kind": "doc",
                       "snippet": "Bands: LOW < 10 <= MEDIUM < 25 <= HIGH < 45 <= CRITICAL"},
                      {"source_id": "tool:compliance_checker+financial_risk_calculator", "title": "Rubric engine", "kind": "tool"}],
        "facts": None, "grounding": {"method": "deterministic", "ok": True, "problems": []},
        "suggestions": ["Which vendors have a knockout?"], "steps": 0, "llm_calls": 0, "tokens": 0, "latency_ms": 12,
        "trace_id": "t1"}


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.messages = []

    def __call__(self, method, url, headers=None, timeout=None, json=None, **kw):
        path = url.split("8000", 1)[-1]
        self.calls.append((method, path, json))
        if method == "POST" and path == "/chat":
            self.messages += [{"id": len(self.messages) + 1, "role": "user", "content": {"text": json["message"]}},
                              {"id": len(self.messages) + 2, "role": "assistant", "content": TURN}]
            return _Resp(201, TURN)
        if path.startswith("/chat/sessions/"):
            return _Resp(200, {"id": "s1", "messages": self.messages})
        if path.startswith("/chat/sessions"):
            return _Resp(200, [{"id": "s1", "title": "Which vendors are rated CRITICAL?"}] if self.messages else [])
        return _Resp(404, {"detail": "not found"})


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def test_chat_view_without_the_live_api_explains_what_is_needed(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[1].set_value("Ask the Copilot").run()
    assert not at.exception
    assert any("runs on the live API" in i.value for i in at.info)
    assert [r.label for r in at.sidebar.radio] == ["Data source", "View"]          # the review view's radio stays first


def test_chat_view_sends_a_question_and_renders_a_grounded_answer(monkeypatch):
    import httpx
    fake = FakeAPI()
    monkeypatch.setattr(httpx, "request", fake)
    monkeypatch.setenv("API_KEY", "human-test-key")
    monkeypatch.setenv("API_URL", "http://localhost:8000")
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[1].set_value("Ask the Copilot").run()
    assert not at.exception
    assert any(b.label == "Which vendors are rated CRITICAL?" for b in at.button)      # example questions
    at.chat_input[0].set_value("Which vendors are rated CRITICAL?").run()
    assert not at.exception
    assert ("POST", "/chat", {"message": "Which vendors are rated CRITICAL?", "session_id": None}) in fake.calls
    assert at.session_state.chat_sid == "s1"
    md = " ".join(m.value for m in at.markdown)
    assert "5 of 35 vendors match" in md and "POLICY-TPRM-001-v3" in md
    assert any("Computed by the copilot's tools" in c.value for c in at.caption)
    assert len(at.dataframe) == 1 and any(e.label == "Sources (2)" for e in at.expander)
