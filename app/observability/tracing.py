"""Tracing for every LLM call and tool call.

    with start_trace("review", input={...}, metadata={"review_id": rid}):      # one trace per request / review
        with span("compliance_checker", kind="tool", input=args) as s:       # one span per tool or model call
            ...
            s.end(output=result)                                             # or s.error(exc)

Two sinks, both best-effort (tracing never breaks a request):
  * Langfuse (v2 SDK) when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST are set: traces, spans, and
    generations with prompt, output, token usage, latency, errors and retries;
  * a local JSONL file (TRACE_LOG_PATH, default observability/traces.jsonl) that always works, for tests and offline use.
TRACING=off disables both. Text fields are truncated to TRACE_MAX_CHARS (default 4000); API keys never pass through here.
"""
from __future__ import annotations

import atexit
import contextvars
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_trace: contextvars.ContextVar = contextvars.ContextVar("pvrc_trace", default=None)
_lock = threading.Lock()
_client = None
_client_failed = False


def _enabled() -> bool:
    return os.getenv("TRACING", "on").lower() not in ("off", "0", "false")


def _log_path() -> Path:
    return Path(os.getenv("TRACE_LOG_PATH", str(ROOT / "observability" / "traces.jsonl")))


def _trim(x, n=None):
    n = n or int(os.getenv("TRACE_MAX_CHARS", "4000"))
    if x is None:
        return None
    if isinstance(x, str):
        return x if len(x) <= n else x[:n] + f"... [+{len(x) - n} chars]"
    try:
        s = json.dumps(x, default=str, ensure_ascii=False)
    except Exception:
        s = str(x)
    return x if len(s) <= n else s[:n] + f"... [+{len(s) - n} chars]"


def langfuse_client():
    """The Langfuse client, or None when not configured / not installed / failed once."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk or "REPLACE" in pk or "REPLACE" in sk:
        _client_failed = True
        return None
    try:
        from langfuse import Langfuse
        _client = Langfuse(public_key=pk, secret_key=sk, host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
                           release=os.getenv("APP_VERSION", "week2"))
        atexit.register(_client.flush)
    except Exception as e:          # never fail the app because tracing is misconfigured
        print(f"[tracing] Langfuse disabled: {type(e).__name__}: {e}")
        _client_failed = True
    return _client


def _write_local(rec: dict):
    try:
        p = _log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock, p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    except Exception:
        pass


class _Trace:
    def __init__(self, name, input=None, metadata=None, trace_id=None, user_id=None, session_id=None):
        self.id = trace_id or uuid.uuid4().hex
        self.name, self.input, self.metadata = name, input, metadata or {}
        self.user_id, self.session_id = user_id, session_id
        self.lf = None
        c = langfuse_client()
        if c is not None:
            try:
                self.lf = c.trace(id=self.id, name=name, input=_trim(input), metadata=self.metadata, user_id=user_id,
                                  session_id=session_id, tags=["pvrc"])
            except Exception:
                self.lf = None


class Span:
    """Handle yielded by span(): call end(...) or error(...); otherwise the span closes with no output."""

    def __init__(self, name, kind, input, metadata, trace: _Trace):
        self.name, self.kind, self.input, self.metadata = name, kind, input, dict(metadata or {})
        self.trace = trace
        self.id = uuid.uuid4().hex[:16]
        self.t0 = time.time()
        self.start = datetime.now(timezone.utc)
        self.output = None
        self.err = None
        self.usage = None
        self.model = None
        self.closed = False

    def end(self, output=None, usage: dict | None = None, model: str | None = None, **metadata):
        self.output, self.usage, self.model = output, usage, model or self.model
        self.metadata.update(metadata)

    def error(self, exc, **metadata):
        self.err = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
        self.metadata.update(metadata)

    def _close(self):
        if self.closed:
            return
        self.closed = True
        latency_ms = int((time.time() - self.t0) * 1000)
        rec = dict(ts=self.t0, trace_id=self.trace.id, trace_name=self.trace.name, span_id=self.id, name=self.name,
                   kind=self.kind, latency_ms=latency_ms, error=self.err, model=self.model, usage=self.usage,
                   metadata=self.metadata, input=_trim(self.input), output=_trim(self.output))
        _write_local(rec)
        lf = self.trace.lf
        if lf is None:
            return
        try:
            end = datetime.now(timezone.utc)
            common = dict(name=self.name, start_time=self.start, end_time=end, input=_trim(self.input), output=_trim(self.output),
                          metadata={**self.metadata, "latency_ms": latency_ms, "kind": self.kind},
                          level="ERROR" if self.err else None, status_message=self.err)
            if self.kind == "generation":
                u = self.usage or {}
                lf.generation(model=self.model, usage_details={"input": int(u.get("prompt_tokens", 0) or 0),
                                                               "output": int(u.get("completion_tokens", 0) or 0)}, **common)
            else:
                lf.span(**common)
        except Exception:
            pass


@contextmanager
def start_trace(name: str, input=None, metadata: dict | None = None, trace_id: str | None = None, user_id: str | None = None,
                session_id: str | None = None):
    """Open a trace for one request; spans opened inside it attach to it. Nested calls reuse the outer trace."""
    if not _enabled() or _trace.get() is not None:
        yield _trace.get()
        return
    t = _Trace(name, input, metadata, trace_id, user_id, session_id)
    token = _trace.set(t)
    try:
        yield t
    finally:
        _trace.reset(token)
        if t.lf is not None:
            try:
                t.lf.update(metadata=t.metadata)
                langfuse_client().flush()
            except Exception:
                pass


def current_trace_id() -> str | None:
    t = _trace.get()
    return t.id if t else None


@contextmanager
def span(name: str, kind: str = "span", input=None, metadata: dict | None = None):
    """A tool call (kind='tool'), model call (kind='generation') or any step (kind='span').
    Outside start_trace(), the span gets a trace of its own, so it is still visible."""
    if not _enabled():
        s = Span(name, kind, input, metadata, _Trace.__new__(_Trace))
        s.closed = True
        yield s
        return
    if _trace.get() is None:
        with start_trace(name, input=input, metadata={"auto": True}):
            with span(name, kind, input, metadata) as s:
                yield s
        return
    s = Span(name, kind, input, metadata, _trace.get())
    try:
        yield s
    except BaseException as e:
        if s.err is None:
            s.error(e)
        raise
    finally:
        s._close()


def read_local(trace_id: str | None = None, limit: int = 200) -> list[dict]:
    p = _log_path()
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if trace_id:
        rows = [r for r in rows if r.get("trace_id") == trace_id]
    return rows[-limit:]
