"""Write app/ui/mock_review.json: a review the UI renders without a running API

    python -m scripts.make_ui_mock                 # stand-in model: free, offline; answers are placeholder text
    python -m scripts.make_ui_mock --live          # your configured model (LLM_* in .env): realistic text, costs tokens
    python -m scripts.make_ui_mock --vendor V031   # another vendor (default V003, a HIGH-risk renewal with an injection)

Runs the real API in-process on a throwaway SQLite database (your store and reviews are not touched): POST /reviews
(full review, through the agent), one follow-up message, then GET /reviews/{id}. So the file has exactly the shape
the live UI gets. A peer comparison is attached for the scorecard tab.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from app.retrieval.chunking import ROOT

load_dotenv()
OUT = ROOT / "app" / "ui" / "mock_review.json"
KEY = "mock-ui-key"


def build(vendor_id: str = "V003", peers: tuple[str, ...] = ("V001", "V002"), live: bool = False) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="pvrc-ui-mock-"))
    env = {"STORE_BACKEND": "sqlite", "STORE_SQLITE_PATH": str(tmp / "mock.sqlite"), "API_KEYS": f"{KEY}:Asha (procurement analyst)",
           "COPILOT_API_KEYS": "", "RETRIEVAL_STRATEGY": "structure", "RETRIEVAL_MODE": "bm25", "RERANK_BACKEND": "none",
           "API_RATE_LIMIT": "1000/minute", "MIGRATE_ON_START": "1", "AGENT_CHECKPOINT_PATH": str(tmp / "cp.sqlite"),
           "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "LLM_CACHE": "0"}
    if not live:
        env["LLM_BACKEND"] = "mock"
    os.environ.update(env)
    from fastapi.testclient import TestClient

    from app.api import main
    from app.db.migrate import upgrade
    from app.db.store import get_store
    from app.retrieval import service
    from app.tools import run_tool
    from scripts import index_corpus
    upgrade()
    s = get_store()
    index_corpus.run(s, ["structure"], verify=False)
    s.close()
    main.State.repo, main.State.store, main.State.graph = None, None, None
    service.reset()
    h = {"X-API-Key": KEY}
    with TestClient(main.app) as c:
        r = c.post("/reviews", headers=h, json={"request_kind": "FULL_REVIEW", "vendor_ids": [vendor_id],
                                                "question": "Prepare the renewal risk review for this vendor."})
        r.raise_for_status()
        rid = r.json()["id"]
        c.post(f"/reviews/{rid}/messages", headers=h, json={"content": "Why is the security risk high?"}).raise_for_status()
        review = c.get(f"/reviews/{rid}", headers=h).json()
    cmp = run_tool("vendor_compare", {"vendor_ids": [*peers, vendor_id]})
    if cmp.ok:
        review["comparison"] = cmp.data
    review["mock_note"] = "stand-in model: answer text is placeholder" if not live else f"model {os.getenv('LLM_MODEL', '?')}"
    return review


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", default="V003")
    ap.add_argument("--live", action="store_true", help="use the configured LLM instead of the stand-in (costs tokens)")
    a = ap.parse_args(argv)
    review = build(a.vendor, live=a.live)
    OUT.write_text(json.dumps(review, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    risk = (review.get("score_json") or {}).get("risk") or {}
    print(f"wrote {OUT.relative_to(ROOT)}: {a.vendor} {risk.get('band')} {risk.get('overall')}, {len(review['turns'])} turns")


if __name__ == "__main__":
    main()
