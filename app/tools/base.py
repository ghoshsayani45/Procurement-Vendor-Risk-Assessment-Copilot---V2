"""Tool framework (step 2.5): every tool is a plain Python function with a Pydantic input and output schema.

    result = run_tool("compliance_checker", {"vendor_id": "V003"})      # -> app.schemas.ToolResult
    result.ok, result.data, result.error, result.latency_ms

run_tool() is the only way agents and the API call tools. It
  1. validates the arguments against the tool's input model (bad arguments -> ok=False, never an exception),
  2. runs the tool with a timeout (TOOL_TIMEOUT_S or the tool's own limit),
  3. validates the result against the output model,
  4. records a span (kind="tool") in the trace and, when an audit sink is given, a TOOL_CALL audit event.
All tools are read-only: none of them writes vendor data, reviews or decisions.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from pydantic import BaseModel, ValidationError

from app.observability.tracing import span
from typing import get_args

from app.schemas import ToolName, ToolResult

DEFAULT_AS_OF = date.fromisoformat(os.getenv("REVIEW_AS_OF", "2026-11-01"))
_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("TOOL_WORKERS", "8")), thread_name_prefix="tool")   # spare threads: a
# timed-out call keeps its thread until it returns, and a queued call's timeout clock already runs while it waits


class ToolError(RuntimeError):
    """A tool's expected failure (unknown vendor, nothing indexed ...): reported as ok=False with this message."""


@dataclass
class ToolContext:
    store: object | None = None                              # app.db.store instance; None = default store
    audit: Callable[[str, dict], None] | None = None         # audit(action, detail) - e.g. the API's audit log
    extras: dict = field(default_factory=dict)

    def get_store(self):
        if self.store is None:
            from app.db.store import get_store
            self.store = get_store()
        return self.store


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    timeout_s: float
    description: str


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, input_model: type[BaseModel], output_model: type[BaseModel], timeout_s: float = 10.0):
    def deco(fn):
        REGISTRY[name] = ToolSpec(name, fn, input_model, output_model, timeout_s, (fn.__doc__ or "").strip())
        return fn
    return deco


def _short_errors(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in err['loc']) or 'input'}: {err['msg']}" for err in e.errors()[:5])


def run_tool(name: str, args: dict | None = None, ctx: ToolContext | None = None) -> ToolResult:
    from app.tools import load_all
    load_all()
    ctx = ctx or ToolContext()
    t0 = time.time()
    spec = REGISTRY.get(name)
    with span(name, kind="tool", input=args) as s:
        def finish(ok, data=None, error=None):
            ms = int((time.time() - t0) * 1000)
            if error:
                s.error(error)
            s.end(output=data if ok else None, ok=ok)
            if ctx.audit:
                try:
                    ctx.audit("TOOL_CALL", {"tool": name, "args": args, "ok": ok, "error": error, "latency_ms": ms})
                except Exception:
                    pass
            if name not in get_args(ToolName):       # unknown name (e.g. invented by a planner): no schema-valid tool field
                return ToolResult.model_construct(tool=name, ok=ok, data=data, error=error, latency_ms=ms)
            return ToolResult(tool=name, ok=ok, data=data, error=error, latency_ms=ms)

        if spec is None:
            return finish(False, error=f"unknown tool {name!r}; available: {', '.join(sorted(REGISTRY))}")
        try:
            inp = spec.input_model.model_validate(args or {})
        except ValidationError as e:
            return finish(False, error=f"invalid arguments: {_short_errors(e)}")
        timeout = float(os.getenv("TOOL_TIMEOUT_S", "0") or 0) or spec.timeout_s
        fut = _POOL.submit(spec.fn, inp, ctx)
        try:
            out = fut.result(timeout=timeout)
        except FutureTimeout:
            fut.cancel()
            return finish(False, error=f"timed out after {timeout:g}s")
        except ToolError as e:
            return finish(False, error=str(e))
        except Exception as e:                       # a bug or an infrastructure failure: report it, don't crash the agent
            return finish(False, error=f"{type(e).__name__}: {e}"[:500])
        try:
            data = spec.output_model.model_validate(out).model_dump(mode="json")
        except ValidationError as e:
            return finish(False, error=f"tool output failed its schema: {_short_errors(e)}")
        return finish(True, data=data)


def describe_tools() -> list[dict]:
    """Name, description and JSON schemas of every tool (for the planner prompt and GET /tools)."""
    from app.tools import load_all
    load_all()
    return [{"name": s.name, "description": s.description, "timeout_s": s.timeout_s,
             "input_schema": s.input_model.model_json_schema(), "output_schema": s.output_model.model_json_schema()}
            for s in REGISTRY.values()]
