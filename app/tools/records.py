"""Vendor records: the structured system of record the tools read.

In a real procurement team these live in the vendor-management system: which certificates are on file with their dates,
the signed contract terms and amendments, financial statements received, SLA performance, debarment and conflict-of-
interest flags. Here they come from data/ground_truth/vendors.json, with the answer-key fields removed, so no tool can
ever see which issues were planted or which traps a vendor carries. Every fact keeps the doc_id of the document that
evidences it, so tool outputs can be cited.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.retrieval.chunking import ROOT

VENDORS_FILE = ROOT / "data" / "ground_truth" / "vendors.json"
ANSWER_KEY_FIELDS = ("planted_issues", "traps", "adversarial", "notes")


def to_record(vendor: dict) -> dict:
    return {k: v for k, v in vendor.items() if k not in ANSWER_KEY_FIELDS}


def load_records_file(path: Path = VENDORS_FILE) -> list[dict]:
    return [to_record(v) for v in json.loads(path.read_text(encoding="utf-8"))]
