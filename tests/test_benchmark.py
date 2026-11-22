"""benchmark statistics (agreement with people, run-to-run spread) and the cache salt."""
from scripts.benchmark import agreement, summarise


def test_agreement_statistics():
    a = agreement([(5, 5), (4, 3), (2, 2), (4, 4), (1, 4)])
    assert a["n"] == 5 and a["exact"] == 0.6 and a["within_1"] == 0.8
    assert a["judge_useful_rate"] == 0.6 and a["human_useful_rate"] == 0.6
    assert a["useful_agreement"] == 0.6 and -1 <= a["cohens_kappa_useful"] <= 1
    perfect = agreement([(5, 5), (2, 2), (4, 4), (1, 1)])
    assert perfect["cohens_kappa_useful"] == 1.0 and perfect["mean_abs_diff"] == 0


def test_spread_and_stability():
    def run(acc, passed):
        return {"metrics": {"accuracy": acc}, "rows": [{"id": i, "passed": p} for i, p in passed.items()]}
    rep = {"runs": [run(0.9, {"Q1": True, "Q2": True, "Q3": False}), run(0.8, {"Q1": True, "Q2": False, "Q3": False})]}
    summarise(rep)
    assert rep["spread"]["accuracy"]["mean"] == 0.85 and rep["spread"]["accuracy"]["sd"] > 0
    assert rep["stability"] == {"always_pass": 1, "never_pass": 1, "flip": ["Q2"]}


def test_cache_salt_changes_keys_only_when_set(monkeypatch):
    from app import llm_client
    msgs = [{"role": "user", "content": "x"}]
    monkeypatch.delenv("LLM_CACHE_SALT", raising=False)
    plain = llm_client._cache_key(msgs, 0, 10, True, None, {})
    monkeypatch.setenv("LLM_CACHE_SALT", "repeat-2")
    assert llm_client._cache_key(msgs, 0, 10, True, None, {}) != plain
    monkeypatch.setenv("LLM_CACHE_SALT", "")
    assert llm_client._cache_key(msgs, 0, 10, True, None, {}) == plain   # unset or empty: the old keys, old cache works


def test_decoding_mode_is_part_of_the_cache_key(monkeypatch):
    from app import llm_client
    msgs = [{"role": "user", "content": "x"}]
    monkeypatch.delenv("LLM_CACHE_SALT", raising=False)
    monkeypatch.setenv("LLM_CONSTRAINED", "json_schema")
    constrained = llm_client._cache_key(msgs, 0, 10, True, {"type": "object"}, {})
    monkeypatch.setenv("LLM_CONSTRAINED", "")
    json_mode = llm_client._cache_key(msgs, 0, 10, True, {"type": "object"}, {})
    monkeypatch.setenv("LLM_CONSTRAINED", "guided_json")
    guided = llm_client._cache_key(msgs, 0, 10, True, {"type": "object"}, {})
    assert len({constrained, json_mode, guided}) == 3            # an ablation never replays another mode's replies
