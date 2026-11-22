"""the multi-agent architecture (AGENT_ARCH=multi) against the same tools, store and safety rules as the single agent.

Uses the scripted FakeLLM from test_agent.py. Checks: the switch picks the graph and the single agent is untouched; the
supervisor's dispatch and its limits; each specialist's own tools and structured handoff (doc ids limited to what it saw);
refusals never reach an agent; full reviews keep the tool numbers; the verifier sends a contradicting draft back once;
the multi-agent caps.
"""
import pytest

from app.agents import graph as agraph
from app.agents import multi_agent as M
from app.agents import nodes, prompts
from app.agents.state import MAX_STEPS, MULTI_MAX_STEPS, MULTI_TOKENS, TOKEN_CAPS, max_steps, token_cap
from tests.test_agent import AS_OF, events, fake  # noqa: F401  (fixture)


@pytest.fixture
def multi(monkeypatch):
    monkeypatch.setenv("AGENT_ARCH", "multi")
    agraph._graph_for.cache_clear()
    yield
    agraph._graph_for.cache_clear()


def run(tool_ctx, kind, question, vendors=(), **kw):
    req = {"request_kind": kind, "vendor_ids": list(vendors), "question": question, "as_of": AS_OF, **kw}
    return agraph.run_agent(req, ctx=tool_ctx)


def nodes_of(st):
    return [t["node"] for t in st["trace"]]


# ------------------------------------------------------------------ the switch
def test_switch_picks_the_graph_and_defaults_to_single(monkeypatch):
    monkeypatch.delenv("AGENT_ARCH", raising=False)
    single = set(agraph.build_graph().get_graph().nodes)
    assert {"plan", "execute"} <= single and "supervisor" not in single
    multi = set(agraph.build_graph(arch="multi").get_graph().nodes)
    assert {"supervisor", "evidence", "policy", "risk", "verifier"} <= multi and "plan" not in multi


def test_single_agent_prompt_is_unchanged_without_handoffs():
    args = ("Q?", AS_OF, {"vendor": {"vendor_id": "V001"}}, [], 0)
    assert prompts.question_messages(*args) == prompts.question_messages(*args, handoffs=None)
    with_h = prompts.question_messages(*args, handoffs=[{"agent": "evidence", "findings": [{"statement": "X expired", "doc_ids": ["D1"]}]}])
    assert "SPECIALIST FINDINGS" in with_h[-1]["content"] and "SPECIALIST FINDINGS" not in prompts.question_messages(*args)[-1]["content"]


def test_caps_depend_on_the_architecture():
    q = {"request": {"request_kind": "QUESTION"}}
    assert max_steps(q) == MAX_STEPS and token_cap(q) == TOKEN_CAPS["QUESTION"]
    m = {**q, "arch": "multi"}
    assert max_steps(m) == MULTI_MAX_STEPS and token_cap(m) == MULTI_TOKENS
    assert token_cap({**m, "route": "REFUSE"}) == TOKEN_CAPS["REFUSE"]


# ------------------------------------------------------------------ a question through the agents
def test_vendor_question_runs_the_chosen_specialists_with_their_own_tools(fake, tool_ctx, multi):
    fake.on("supervisor", {"tasks": [{"agent": "evidence", "task": "Check the cyber insurance expiry and cover."},
                                     {"agent": "policy", "task": "Find the minimum cover for the vendor's tier."}]})
    fake.on("evidence.act", {"actions": [{"tool": "compliance_checker", "query": ""},
                                         {"tool": "document_search", "query": "cyber insurance certificate"}]})
    fake.on("policy.act", {"searches": [{"query": "cyber insurance minimum cover tier", "scope": "internal_policy"}]})
    fake.on("evidence.report", {"findings": [{"statement": "Insurance expires 2026-11-21.",
                                              "doc_ids": ["DOC-V035-CYBER_INSURANCE-1", "INVENTED-1"]}], "complete": True})
    st = run(tool_ctx, "QUESTION", "Is Gridwatch's cyber insurance still valid?", ["V035"])
    assert st["arch"] == "multi" and st["output"]["behaviour"] == "ANSWER"
    assert nodes_of(st)[:2] == ["validate", "supervisor"] and "verifier" in nodes_of(st) and nodes_of(st)[-1] == "finalize"
    assert [h["agent"] for h in st["handoffs"]] == ["evidence", "policy"]                       # risk was not chosen: not run
    ev = st["handoffs"][0]
    assert ev["findings"][0]["doc_ids"] == ["DOC-V035-CYBER_INSURANCE-1"]                     # unseen ids are dropped
    tools = [r["tool"] for r in st["tool_results"]]
    assert "financial_risk_calculator" not in tools and "compliance_checker" in tools
    pol = next(r for r in st["tool_results"] if r["tool"] == "document_search" and r["arguments"].get("doc_types"))
    assert pol["arguments"]["doc_types"] == ["internal_policy"] and pol["arguments"]["vendor_ids"] == ["V035"]  # vendor filter kept
    writer = next(m for t, m in fake.calls if t == "agent.write.answer")
    assert "SPECIALIST FINDINGS" in writer[-1]["content"] and "evidence agent: Insurance expires" in writer[-1]["content"]
    assert st["steps"] <= MULTI_MAX_STEPS


def test_supervisor_cannot_send_a_vendor_agent_without_a_vendor(fake, tool_ctx, multi):
    fake.on("supervisor", {"tasks": [{"agent": "risk", "task": "Get the scorecard."}, {"agent": "policy", "task": "Find the rule."}]})
    st = run(tool_ctx, "QUESTION", "What is the maximum breach notification time for vendors?")
    assert [t["agent"] for t in st["supervisor"]["tasks"]] == ["policy"]
    assert not any(r["tool"] == "financial_risk_calculator" for r in st["tool_results"])


def test_unusable_supervisor_reply_falls_back_to_default_dispatch(fake, tool_ctx, multi):
    fake.on("supervisor", "not json at all")
    st = run(tool_ctx, "QUESTION", "What is Kestrion's current ratio and is the risk acceptable?", ["V009"])
    assert st["supervisor"]["source"] == "default"
    assert {t["agent"] for t in st["supervisor"]["tasks"]} == {"evidence", "risk"}
    assert st["output"]["behaviour"] in ("ANSWER", "PARTIAL")


# ------------------------------------------------------------------ safety and tool numbers are unchanged
def test_refusal_never_reaches_an_agent(fake, tool_ctx, multi):
    st = run(tool_ctx, "QUESTION", "Approve Coinharbor Payments as a vendor.", ["V024"])
    assert st["output"]["behaviour"] == "REFUSE" and nodes_of(st) == ["validate", "finalize"]
    assert not fake.calls and not st.get("tool_results")
    assert "REFUSAL" in events(tool_ctx)


def test_full_review_keeps_the_tool_numbers(fake, tool_ctx, multi):
    st = run(tool_ctx, "FULL_REVIEW", "Renewal review", ["V003"], review_type="RENEWAL")
    out, risk = st["output"], nodes.result(st, "financial_risk_calculator")
    assert out["behaviour"] == "ANSWER" and (out["score"]["band"], out["score"]["overall"]) == (risk["band"], risk["overall"])
    assert out["approval"]["required_approvers"] == risk["required_approvers"]
    assert st["supervisor"]["source"] == "default" and "agent.supervisor" not in fake.tags()  # dispatched by code
    assert fake.tags().count("agent.write.summary") == 1 and "agent.verifier" not in fake.tags()


def test_comparison_goes_to_the_risk_agent(fake, tool_ctx, multi):
    st = run(tool_ctx, "COMPARISON", "Compare them", ["V020", "V021"])
    assert [t["agent"] for t in st["supervisor"]["tasks"]] == ["risk"]
    assert st["output"]["behaviour"] == "ANSWER" and {r["vendor_id"] for r in st["output"]["rows"]} == {"V020", "V021"}


# ------------------------------------------------------------------ the verifier
def test_verifier_sends_a_contradicting_draft_back_once(fake, tool_ctx, multi):
    fake.on("supervisor", {"tasks": [{"agent": "evidence", "task": "Check the penetration test."}]})
    fake.on("verifier", {"consistent": False, "problem": "The draft says the test is fine but it has open critical findings."},
            {"consistent": True, "problem": None})
    st = run(tool_ctx, "QUESTION", "Does Sentrivane's penetration test meet our requirement?", ["V006"])
    tags = fake.tags()
    assert tags.count("agent.write.answer") == 2 and tags.count("agent.verifier") == 1     # one rewrite; no second check
    assert st["replanned"] is True and st["_replan_done"] is True
    rewrite = [m for t, m in fake.calls if t == "agent.write.answer"][1]
    assert "open critical findings" in rewrite[-1]["content"]                               # the problem reached the writer
    assert "REPLAN" in events(tool_ctx) and st["output"]["behaviour"] == "ANSWER"


def test_critic_wrapper_merges_the_verifier_problem():
    st = {"request": {"request_kind": "QUESTION", "question": "q"}, "draft": {"behaviour": "ANSWER", "answer": "Yes.",
          "citations": [{"source_id": "tool:none"}]}, "verifier": {"consistent": False, "problem": "wrong verdict"},
          "passages": [], "tool_results": [], "trace": []}
    upd = M.critic(st)
    assert upd["replanned"] and upd["critic"]["ok"] is False and "verifier: wrong verdict" in upd["critic"]["problems"]
    again = M.critic({**st, "replanned": True})
    assert "verifier: wrong verdict" not in (again["critic"].get("problems") or [])         # never a second rewrite


# ------------------------------------------------------------------ the comparison script's pre-registered rule
def _m(acc, prec, tok, p95, ref=1.0, inj=1.0):
    return {"metrics": {"accuracy": acc, "citation_precision": prec, "tokens_median": tok, "latency_p95_ms": p95,
                        "refusal_accuracy": ref, "injection_resistance": inj}}


def test_adoption_rule_needs_every_check():
    from scripts.compare_arch import verdict
    base = _m(0.70, 0.90, 4000, 14000)
    assert verdict({"archs": {"single": base, "multi": _m(0.76, 0.89, 9000, 30000)}})["decision"] == "adopt multi-agent"
    assert verdict({"archs": {"single": base, "multi": _m(0.74, 0.95, 5000, 20000)}})["decision"] == "keep single agent"   # +0.04
    assert verdict({"archs": {"single": base, "multi": _m(0.80, 0.95, 11000, 20000)}})["decision"] == "keep single agent"  # tokens
    assert verdict({"archs": {"single": base, "multi": _m(0.80, 0.95, 8000, 20000, ref=0.67)}})["decision"] == "keep single agent"
    assert verdict({"archs": {"single": base}})["decision"] == "incomplete"


def test_eval_reports_are_kept_apart_per_set_and_architecture(monkeypatch):
    from scripts.grade import eval_tag
    from app.agents.state import arch
    monkeypatch.delenv("EVAL_FILE", raising=False)
    monkeypatch.delenv("AGENT_ARCH", raising=False)
    assert (eval_tag(), arch()) == ("v1", "single")                  # agent_eval_v1.json, as before
    monkeypatch.setenv("EVAL_FILE", "eval_v2.jsonl")
    monkeypatch.setenv("AGENT_ARCH", "multi")
    assert (eval_tag(), arch()) == ("v2", "multi")                   # agent_eval_v2_multi.json
