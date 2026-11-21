"""Week 3 (3.3): the reliability layer around every LLM call the agent makes.

call_json(state, messages, model) -> (parsed model instance or None, state updates)
  1. constrained decoding when LLM_CONSTRAINED is set (the schema goes to the endpoint),
  2. otherwise JSON mode,
  3. cheap repair: code fences, trailing commas, single quotes, the first {...} block,
  4. one retry with the validation error in the prompt,
  5. give up -> None: the caller uses its safe default and marks the stage 'defaulted'.
Before each call: the step cap and the per-request token budget are checked; prompts that cannot fit the context
window raise ContextTooLong so the caller can shrink them. Every outcome is counted in state["json_stats"].
"""
from __future__ import annotations

import json
import os
import re

from pydantic import BaseModel, ValidationError

from app import llm_client
from app.agents.state import MAX_STEPS, MULTI_MAX_STEPS, token_cap


class BudgetExceeded(RuntimeError):
    pass


def _repair(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        t = m.group(0)
    t = re.sub(r",\s*([}\]])", r"\1", t)                              # trailing commas
    if "'" in t and '"' not in t:                                     # single-quoted JSON
        t = t.replace("'", '"')
    return t


def _parse_strict(text: str, model: type[BaseModel]):
    """No repair: the reply must be valid JSON for the schema as it comes (AGENT_JSON_RETRY=0 ablation)."""
    try:
        return model.model_validate(json.loads((text or "").strip())), None, False
    except (json.JSONDecodeError, ValidationError) as e:
        return None, str(e)[:600], False


def _parse(text: str, model: type[BaseModel]):
    """Returns (instance, error, repaired?)."""
    for repaired, candidate in ((False, (text or "").strip()), (True, _repair(text))):
        try:
            return model.model_validate(json.loads(candidate)), None, repaired
        except (json.JSONDecodeError, ValidationError) as e:
            err = str(e)[:600]
    return None, err, False


def _account(state: dict, upd: dict):
    rec = llm_client.LAST_CALL or {}
    upd["steps"] = upd.get("steps", state.get("steps", 0)) + 1
    upd["llm_calls"] = upd.get("llm_calls", state.get("llm_calls", 0)) + 1
    upd["tokens"] = upd.get("tokens", state.get("tokens", 0)) + int(rec.get("total_tokens", 0) or 0)
    if rec.get("cached"):                       # replayed from the response cache: count it, and its original latency
        upd["cached_llm_calls"] = upd.get("cached_llm_calls", state.get("cached_llm_calls", 0)) + 1
        upd["cached_latency_ms"] = upd.get("cached_latency_ms", state.get("cached_latency_ms", 0)) + int(rec.get("latency_ms") or 0)
        if rec.get("usage_estimated") or not rec.get("latency_known"):
            upd["cached_estimated"] = True


def check_budget(state: dict, upd: dict, estimate: int):
    steps = upd.get("steps", state.get("steps", 0))
    tokens = upd.get("tokens", state.get("tokens", 0))
    limit = MULTI_MAX_STEPS if state.get("arch") == "multi" else MAX_STEPS
    if steps + 1 > limit:
        raise BudgetExceeded(f"step cap reached ({limit} steps)")
    cap = token_cap(state)
    if tokens + estimate > cap:
        raise BudgetExceeded(f"token budget reached ({tokens:,} used, next call ~{estimate:,}, cap {cap:,})")


def call_json(state: dict, messages: list[dict], model: type[BaseModel], tag: str, max_tokens: int = 600,
              schema: dict | None = None):
    """One schema-validated LLM call with repair and one retry. Returns (instance | None, state updates).
    `schema` overrides the JSON schema sent for constrained decoding (e.g. citations limited to an enum of real ids);
    the reply is still validated against `model`."""
    upd: dict = {"json_stats": dict(state.get("json_stats") or {})}
    js = upd["json_stats"]
    js["calls"] = js.get("calls", 0) + 1
    schema = schema or model.model_json_schema()
    constrained = bool(os.getenv("LLM_CONSTRAINED", "").strip())
    msgs = list(messages)
    retries = os.getenv("AGENT_JSON_RETRY", "1") == "1"      # 0 = ablation: no repair, no retry (first reply or default)
    for attempt in ((1, 2) if retries else (1,)):
        check_budget(state, upd, llm_client.estimate_prompt_tokens(msgs) + max_tokens)
        try:
            raw = llm_client.chat(msgs, tag=f"agent.{tag}", max_tokens=max_tokens, json_mode=True, schema=schema)
        finally:
            _account(state, upd)
        obj, err, repaired = _parse(raw, model) if retries else _parse_strict(raw, model)
        if obj is not None:
            if attempt == 1 and not repaired:
                js["first_try"] = js.get("first_try", 0) + 1
                if constrained:
                    js["constrained"] = js.get("constrained", 0) + 1
            elif repaired and attempt == 1:
                js["repaired"] = js.get("repaired", 0) + 1
            else:
                js["retry_succeeded"] = js.get("retry_succeeded", 0) + 1
            return obj, upd
        if attempt == 1 and retries:                          # a retry follows only when retries are on
            js["retried"] = js.get("retried", 0) + 1
        msgs = msgs + [{"role": "assistant", "content": (raw or "")[:1500]},
                       {"role": "user", "content": f"That was not valid. Error: {err[:400]}\nReturn ONLY one JSON object with "
                                                   f"exactly these fields: {', '.join(schema.get('properties', {}))}."}]
    js["defaulted"] = js.get("defaulted", 0) + 1
    return None, upd
