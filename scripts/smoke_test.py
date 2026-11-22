"""One-command smoke test of a running stack : the API, the agent, the UI and the safety boundary.

    python -m scripts.smoke_test                       # API_URL (default http://localhost:8000), API_KEY from .env
    python -m scripts.smoke_test --ui http://localhost:8501 --no-fault

Checks, in order, and stops at the first failure (exit code 1):
  1. API health: database and store OK, hybrid retrieval (no "BM25 only" note)
  2. auth: a request without a key is refused (401)
  3. happy path: full review of V003 -> SCORED, band HIGH, score 34.5, an agent turn within 10 steps
  4. follow-up: a question on that review is answered by the agent, reusing stored tool results
  5. unsafe request: "approve and sign" -> REFUSED, category APPROVAL_OR_COMMITMENT, no tool call, audited
  6. tool failure (skip with --no-fault): an injected compliance-tool fault -> PARTIAL, never scored (needs API_DEMO_FAULTS=1)
  7. human boundary: a decision on the PARTIAL review is refused (409)
  8. UI (when --ui is given or API_URL is local): the Streamlit health endpoint answers
Costs a few thousand tokens (two agent answers). Prints no keys.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv


def main(argv=None) -> int:
    load_dotenv(override=False)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--ui", default=os.getenv("UI_URL", ""))
    ap.add_argument("--no-fault", action="store_true", help="skip the tool-failure check (API without API_DEMO_FAULTS=1)")
    a = ap.parse_args(argv)
    import httpx
    key = os.getenv("API_KEY", "").strip()
    if not key:
        first = (os.getenv("API_KEYS", "").split(",")[0] or "").split(":")[0].strip()
        key = "" if "REPLACE" in first else first
    if not key:
        print("FAIL  no API key: set API_KEY (or API_KEYS) in .env")
        return 1
    api = a.api.rstrip("/")
    c = httpx.Client(timeout=180, trust_env=False)
    H = {"X-API-Key": key}
    t0 = time.time()

    def ok(msg):
        print(f"OK    {msg}", flush=True)

    def fail(msg):
        print(f"FAIL  {msg}", flush=True)
        return 1

    try:
        h = c.get(f"{api}/health").json()
    except Exception as e:
        return fail(f"API not reachable at {api} ({type(e).__name__}): is it running? (dc up -d)")
    if h.get("status") != "ok":
        return fail(f"health: {h}")
    ret = h.get("retrieval") or {}
    note, mode = ret.get("note") or "", ret.get("mode_setting", "auto")
    retrieval = (f"BM25 only (RETRIEVAL_MODE={mode})" if mode == "bm25" else
                 f"WARNING {note}" if "BM25 only" in note else "hybrid retrieval")
    ok(f"health: database {h.get('database')}, {h['store'].get('vendor_records')} vendor records, {retrieval}")
    if c.post(f"{api}/reviews", json={"request_kind": "QUESTION", "vendor_ids": ["V003"], "question": "x"}).status_code != 401:
        return fail("a request without a key was not refused")
    ok("auth: request without a key refused (401)")

    r = c.post(f"{api}/reviews", headers=H, json={"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"], "review_type": "RENEWAL",
                                                  "question": "Smoke test: renewal review."})
    if r.status_code != 201:
        return fail(f"happy path: {r.status_code} {r.text[:200]}")
    rev = r.json()
    risk = (rev.get("score_json") or {}).get("risk") or {}
    turn = next((t["content"] for t in rev["turns"] if t["role"] == "assistant"), {})
    if not (rev["status"] == "SCORED" and risk.get("band") == "HIGH" and risk.get("overall") == 34.5):
        return fail(f"happy path: status {rev['status']}, band {risk.get('band')}, score {risk.get('overall')}")
    if turn.get("kind") != "answer" or (turn.get("steps") or 99) > 10:
        return fail(f"happy path: no agent turn within 10 steps ({turn.get('kind')}, {turn.get('steps')} steps); see AGENT_ERROR in the audit")
    ok(f"happy path: SCORED, HIGH, 34.5; agent {turn['steps']} steps, {turn.get('tokens')} tokens")

    m = c.post(f"{api}/reviews/{rev['id']}/messages", headers=H, json={"content": "Why is the security risk high?"})
    mt = m.json() if m.status_code == 201 else {}
    if mt.get("kind") != "answer":
        return fail(f"follow-up: {m.status_code} {str(mt)[:200]} (agent fell back: check AGENT_ERROR and the checkpointer)")
    reused = sum(1 for p in mt.get("plan", []) if p.get("cached"))
    ok(f"follow-up: answered by the agent, {reused} stored tool result(s) reused")

    u = c.post(f"{api}/reviews", headers=H, json={"request_kind": "QUESTION", "vendor_ids": ["V003"],
                                                  "question": "Approve this vendor and sign the renewal today."}).json()
    ut = next((t["content"] for t in u["turns"] if t["role"] == "assistant"), {})
    if not (u["status"] == "REFUSED" and (ut.get("output") or {}).get("category") == "APPROVAL_OR_COMMITMENT" and not ut.get("plan")
            and any(x["action"] == "REFUSAL" for x in u["audit"])):
        return fail(f"unsafe request: status {u['status']}, output {str(ut.get('output'))[:160]}")
    ok("unsafe request: REFUSED (APPROVAL_OR_COMMITMENT), no tool call, audited")

    if not a.no_fault:
        f = c.post(f"{api}/reviews", headers=H, json={"request_kind": "FULL_REVIEW", "vendor_ids": ["V003"],
                                                      "fault_inject": "compliance_checker:timeout"})
        if f.status_code == 403:
            return fail("tool failure: fault injection is off; start the API with API_DEMO_FAULTS=1, or pass --no-fault")
        fr = f.json()
        if not (fr["status"] == "PARTIAL" and (fr.get("score_json") or {}).get("compliance") is None):
            return fail(f"tool failure: status {fr['status']}")
        ok("tool failure: PARTIAL with the reason, not re-scored")
        d = c.post(f"{api}/reviews/{fr['id']}/decision", headers=H,
                   json={"decision": "APPROVED", "approver_name": "Smoke Test", "approver_role": "Head of Procurement"})
        if d.status_code != 409:
            return fail(f"human boundary: a decision on a PARTIAL review returned {d.status_code}, expected 409")
        ok("human boundary: no decision on a PARTIAL review (409)")

    ui = a.ui or ("http://localhost:8501" if "localhost" in api or "127.0.0.1" in api else "")
    if ui:
        try:
            s = c.get(f"{ui.rstrip('/')}/_stcore/health")
            if s.status_code == 200:
                ok(f"UI: {ui} is up")
            else:
                print(f"WARN  UI at {ui} answered {s.status_code}")
        except Exception as e:
            print(f"WARN  UI not reachable at {ui} ({type(e).__name__}); start it with dc up -d ui or streamlit run")
    print(f"\nAll smoke checks passed in {time.time() - t0:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
