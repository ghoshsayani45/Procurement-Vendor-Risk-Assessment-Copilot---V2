"""eval_v2 : frozen before the multi-agent work, well-formed, consistent with the ground truth, and gradable."""
import hashlib
import json
from collections import Counter

import pytest

from scripts import grade as G
from tests.test_eval_set import ROOT, EV, EXP, oracle


@pytest.fixture()
def v2(monkeypatch):
    monkeypatch.setenv("EVAL_FILE", "eval_v2.jsonl")
    return G.load_eval()


def test_eval_v2_is_frozen():
    expected = (EV / "eval_v2.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256((EV / "eval_v2.jsonl").read_bytes()).hexdigest() == expected


def test_eval_file_switch_defaults_to_v1(monkeypatch):
    monkeypatch.delenv("EVAL_FILE", raising=False)
    assert G.eval_path().name == "eval_v1.jsonl" and G.eval_tag() == "v1"
    monkeypatch.setenv("EVAL_FILE", "eval_v2.jsonl")
    assert G.eval_path().name == "eval_v2.jsonl" and G.eval_tag() == "v2"


def test_shape_and_no_overlap_with_v1(v2):
    assert len(v2) == 45 and len({q["id"] for q in v2}) == 45 and {q["split"] for q in v2} == {"test"}
    v1 = {json.loads(l)["request"]["question"].lower() for l in open(EV / "eval_v1.jsonl", encoding="utf-8")}
    assert not [q["id"] for q in v2 if q["request"]["question"].lower() in v1]
    assert Counter(q["expected_behaviour"] for q in v2) == {"ANSWER": 40, "REFUSE": 3, "CLARIFY": 2}


def test_gold_quotes_and_reviews_match_ground_truth(v2):
    import re
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    corpus = {json.loads(l)["doc_id"]: norm(json.loads(l)["text"]) for l in open(ROOT / "data/processed/corpus_v1.jsonl", encoding="utf-8")}
    for q in v2:
        assert set(q["gold_doc_ids"]) <= set(corpus), q["id"]
        for g in q["gold_quotes"]:
            assert norm(g["quote"]) in corpus[g["doc_id"]], (q["id"], g["doc_id"])
        if q["category"] == "full_review":
            o = EXP[q["request"]["vendor_ids"][0]]
            assert (q["expected"]["band"], q["expected"]["overall"]) == (o["band"], o["overall"])


def test_oracle_scores_100_percent_and_naive_fails(v2):
    s, results = G.grade({q["id"]: oracle(q) for q in v2})
    assert s["accuracy"] == 1.0, [r["id"] for r in results if not r["passed"]]
    naive = {q["id"]: {"behaviour": "ANSWER", "answer": "Yes, low risk, proceed.", "score": {"band": "LOW", "overall": 0},
                       "recommendation": "PROCEED", "findings": [], "checklist": [], "rows": [],
                       "approval": {"required_approvers": [], "human_decision": "APPROVED"}} for q in v2}
    s, _ = G.grade(naive)
    assert s["accuracy"] < 0.15 and s["refusal_accuracy"] == 0.0
