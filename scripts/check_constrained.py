"""Day-1 check (Week 3): does the course endpoint support constrained decoding (schema-valid JSON by construction)?

    python -m scripts.check_constrained

Tries, with a tiny schema and ~60 tokens per attempt:
  1. response_format = {"type": "json_schema", ...}   (OpenAI-style structured output; vLLM >= 0.6)
  2. extra_body = {"guided_json": schema}             (vLLM guided decoding)
  3. response_format = {"type": "json_object"}        (plain JSON mode, the Week 1 setting)
and prints which ones the server accepts AND obeys. Put the best one in .env as LLM_CONSTRAINED=json_schema or
LLM_CONSTRAINED=guided_json (leave it empty to use JSON mode + validation + retry), and record it as a decision.
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

SCHEMA = {"type": "object", "properties": {"band": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                                           "reason": {"type": "string", "maxLength": 80}},
          "required": ["band", "reason"], "additionalProperties": False}
MESSAGES = [{"role": "user", "content": "A vendor scored 34.5 on a 0-100 risk scale where HIGH starts at 25 and CRITICAL at 45. "
                                        "Give the band and a short reason. Reply as JSON."}]


def attempt(mode: str):
    import app.llm_client as lc
    os.environ["LLM_CONSTRAINED"] = mode if mode != "json_object" else ""
    os.environ["LLM_CACHE"] = "0"
    try:
        out = lc.chat(MESSAGES, tag=f"check_constrained.{mode}", max_tokens=60, json_mode=True,
                      schema=SCHEMA if mode != "json_object" else None)
    except Exception as e:
        return False, f"rejected: {str(e)[:160]}"
    try:
        obj = json.loads(out)
    except json.JSONDecodeError:
        return False, f"accepted, but output is not JSON: {out[:120]!r}"
    ok = set(obj) == {"band", "reason"} and obj["band"] in SCHEMA["properties"]["band"]["enum"]
    return ok, ("obeys the schema" if ok else "accepted, but output does not follow the schema") + f": {json.dumps(obj)[:120]}"


def main():
    if os.getenv("LLM_BACKEND", "endpoint") == "mock":
        raise SystemExit("LLM_BACKEND=mock: run this against the real endpoint")
    results = {m: attempt(m) for m in ("json_schema", "guided_json", "json_object")}
    for m, (ok, msg) in results.items():
        print(f"{m:12s} {'OK ' if ok else 'NO '} {msg}")
    best = next((m for m in ("json_schema", "guided_json") if results[m][0]), None)
    print("\nRecommendation:", f"LLM_CONSTRAINED={best}" if best else
          "constrained decoding not available: keep LLM_CONSTRAINED empty (JSON mode + validation + retry)")


if __name__ == "__main__":
    main()
