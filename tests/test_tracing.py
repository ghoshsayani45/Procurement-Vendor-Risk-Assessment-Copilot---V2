"""Step 2.7: tracing of LLM and tool calls (local JSONL sink always; Langfuse sink checked with a fake SDK client)."""
from __future__ import annotations

import pytest

import app.observability.tracing as tr
from app.observability.tracing import read_local, span, start_trace


def test_spans_share_the_request_trace_and_record_errors():
    with start_trace("review.create", input={"vendor": "V001"}) as t:
        with span("compliance_checker", kind="tool", input={"vendor_id": "V001"}) as s:
            s.end(output={"tier": 1})
        with pytest.raises(ValueError):
            with span("vendor_compare", kind="tool", input={}):
                raise ValueError("bad input")
    rows = read_local(t.id)
    assert [r["name"] for r in rows] == ["compliance_checker", "vendor_compare"]
    assert rows[0]["output"] == {"tier": 1} and rows[0]["error"] is None and rows[0]["latency_ms"] >= 0
    assert rows[1]["error"] == "ValueError: bad input"


def test_span_outside_a_trace_gets_its_own():
    with span("document_search", kind="tool", input={"query": "q"}) as s:
        s.end(output=[])
    assert read_local()[-1]["trace_name"] == "document_search"


def test_llm_calls_are_traced_as_generations(monkeypatch):
    from app import llm_client
    monkeypatch.setenv("LLM_BACKEND", "mock")
    llm_client.chat([{"role": "user", "content": "QUESTION: What is V001's tier?"}], tag="answer")
    g = read_local()[-1]
    assert g["kind"] == "generation" and g["name"] == "answer" and g["model"] == "mock"
    assert set(g["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"} and g["metadata"]["attempts"] == 1


def test_cached_llm_calls_are_traced_with_zero_tokens(monkeypatch, tmp_path):
    from app import llm_client
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path / "cache")
    msgs = [{"role": "user", "content": "QUESTION: What is V001's tier?"}]
    first = llm_client.chat(msgs, tag="answer")
    assert llm_client.chat(msgs, tag="answer") == first and llm_client.LAST_CALL["cached"]
    g = read_local()[-1]
    assert g["metadata"]["cached"] and g["metadata"]["attempts"] == 0 and g["usage"]["total_tokens"] == 0   # not billed
    assert llm_client.LAST_CALL["total_tokens"] > 0 and not llm_client.LAST_CALL["usage_estimated"]   # cost replayed for metrics


def test_tracing_off_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACING", "off")
    with start_trace("x"):
        with span("y", input=1) as s:
            s.end(output=2)
    assert read_local() == []


def test_long_payloads_are_truncated(monkeypatch):
    monkeypatch.setenv("TRACE_MAX_CHARS", "100")
    with span("big", input="x" * 1000) as s:
        s.end(output={"text": "y" * 1000})
    r = read_local()[-1]
    assert len(r["input"]) < 200 and "[+900 chars]" in r["input"] and isinstance(r["output"], str)


class FakeLangfuse:
    instances = []

    def __init__(self, **kw):
        self.kw, self.calls = kw, []
        FakeLangfuse.instances.append(self)

    def trace(self, **kw):
        self.calls.append(("trace", kw))
        return FakeObs(self)

    def flush(self):
        self.calls.append(("flush", {}))


class FakeObs:
    def __init__(self, root):
        self.root = root

    def span(self, **kw):
        self.root.calls.append(("span", kw))

    def generation(self, **kw):
        self.root.calls.append(("generation", kw))

    def update(self, **kw):
        self.root.calls.append(("update", kw))


def test_langfuse_receives_trace_spans_and_generations(monkeypatch):
    import langfuse
    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse:3000")
    monkeypatch.setattr(tr, "_client", None)
    monkeypatch.setattr(tr, "_client_failed", False)
    with start_trace("review.create", input={"v": 1}, user_id="asha"):
        with span("vendor_search", kind="tool", input={"query": "x"}) as s:
            s.end(output={"resolution": "unique"})
        with span("answer", kind="generation", input=[{"role": "user", "content": "hi"}]) as g:
            g.end(output="hello", model="qwen3-8b", usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})
    lf = FakeLangfuse.instances[-1]
    kinds = [c[0] for c in lf.calls]
    assert kinds[0] == "trace" and "span" in kinds and "generation" in kinds and kinds[-1] == "flush"
    assert lf.kw["host"] == "http://langfuse:3000"
    gen = dict(lf.calls)["generation"]
    assert gen["model"] == "qwen3-8b" and gen["usage_details"] == {"input": 12, "output": 3} and gen["output"] == "hello"
    sp = dict(lf.calls)["span"]
    assert sp["name"] == "vendor_search" and sp["metadata"]["kind"] == "tool" and sp["level"] is None


def test_langfuse_failure_never_breaks_the_call(monkeypatch):
    import langfuse

    def boom(**kw):
        raise RuntimeError("cannot reach langfuse")
    monkeypatch.setattr(langfuse, "Langfuse", boom)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(tr, "_client", None)
    monkeypatch.setattr(tr, "_client_failed", False)
    with span("tool", kind="tool") as s:
        s.end(output=1)
    assert read_local()[-1]["name"] == "tool"


def test_rejected_schema_steps_down_to_json_mode(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app import llm_client
    sent = []

    class Rejected(Exception):
        status_code = 400

    class FakeCompletions:
        def create(self, **kw):
            sent.append(kw.get("response_format"))
            if (kw.get("response_format") or {}).get("type") == "json_schema":
                raise Rejected("unsupported keyword 'enum' in schema")
            msg = SimpleNamespace(content='{"answer": "ok", "citations": ["D1"]}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    monkeypatch.setattr(llm_client, "_client", lambda env: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    monkeypatch.setattr(llm_client, "LEDGER", tmp_path / "ledger.jsonl")
    for k, v in {"LLM_BACKEND": "endpoint", "LLM_MODEL": "qwen3-8b", "LLM_CONSTRAINED": "json_schema", "LLM_CACHE": "0"}.items():
        monkeypatch.setenv(k, v)
    out = llm_client.chat([{"role": "user", "content": "q"}], tag="t", json_mode=True,
                          schema={"type": "object", "properties": {"citations": {"type": "array", "items": {"enum": ["D1"]}}}})
    assert out.startswith('{"answer"') and [f["type"] for f in sent] == ["json_schema", "json_object"]
    assert llm_client.LAST_CALL["schema_fallback"] is True
