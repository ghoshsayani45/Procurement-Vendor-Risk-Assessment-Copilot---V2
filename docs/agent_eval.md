"""Week 3 (3.8, 3.9): run the agent over eval_v1, grade it with the Week 1 grader, report the template's section V metrics
and the four-class failure taxonomy.

    python -m scripts.agent_eval --limit 10                # daily smoke run (dev): reports/smoke/
    python -m scripts.agent_eval --split dev               # full dev run: reports/agent_eval_v1.json + docs/agent_eval.md
    python -m scripts.agent_eval --split test              # ONCE, at the Week 3 gate (D12); merged into the same report
    python -m scripts.agent_eval --ids EVAL-007,EVAL-069   # specific queries (smoke)
    FAULT_INJECT=document_search:timeout python -m scripts.agent_eval --ids EVAL-025   # tool-failure path

Metrics (template section V): end-to-end accuracy, task completion rate, average tool calls per task, tokens (cost)
and latency per query (p50 / p95), failure taxonomy. Plus the grader's detailed checks and the reliability counters
(JSON first try / repaired / retried / defaulted, constrained decoding, repeated calls prevented, replans).
Set LLM_CACHE=1 to make reruns of unchanged queries free. With LLM_BACKEND=mock results go to reports/mock/.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter

from dotenv import load_dotenv

from app import llm_client
from app.agents.graph import run_agent
from app.agents.state import MAX_STEPS, token_cap
from app.agents.taxonomy import CLASSES, primary_class
from app.failure_analysis import classify
from app.retrieval.chunking import ROOT
from app.tools.base import ToolContext
from scripts.grade import grade, load_eval

load_dotenv()
WEEK1_DEV = {   # Week 1 single-prompt baselines on dev (docs/baselines.md): few-shot / CoT-SC / decomposition
    "accuracy": (0.5, 0.533, 0.5), "band_accuracy": (0.43, 0.43, 0.43), "checklist_accuracy": (0.79, 0.85, 0.60),
    "finding_recall": (0.14, 0.14, 0.12), "recommendation_accuracy": (0.29, 0.29, 0.57), "fact_recall": (0.66, 0.73, 0.78),
    "refusal_accuracy": (1.0, 1.0, 1.0), "injection_resistance": (0.0, 0.0, 0.0), "tokens_per_query": (4700, 14900, 8400),
}


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def warm_up():
    """Load the retrieval indexes and the embedding model before timing anything. On a laptop the first
    document_search (BM25 build + 14k vectors + bge-small) can take longer than its 30 s tool timeout, which would
    turn the first query into a PARTIAL and inflate its latency."""
    from app.tools.base import run_tool
    t0 = time.time()
    print("warming up retrieval (first load can take a minute)...", flush=True)
    r = run_tool("document_search", {"query": "warm up retrieval index", "k": 1}, ToolContext())
    if not r.ok:                                              # the timeout does not stop the load: wait once more
        r = run_tool("document_search", {"query": "warm up retrieval index", "k": 1}, ToolContext())
    print(f"retrieval ready in {time.time() - t0:.0f}s" + ("" if r.ok else f" (still failing: {r.error})"), flush=True)


def run_one(q: dict) -> dict:
    req = dict(q["request"])
    req["as_of"] = str(req.get("as_of") or "2026-11-01")
    from app.retrieval.retrievers import DENSE_FALLBACKS
    fb0 = len(DENSE_FALLBACKS)
    t0 = time.time()
    try:
        st = run_agent(req, review_id="", ctx=ToolContext())
    except Exception as e:                                   # a crash is a tool-call error in the taxonomy, never a silent skip
        return dict(id=q["id"], output=None, crashed=f"{type(e).__name__}: {e}"[:300], latency_ms=int((time.time() - t0) * 1000),
                    steps=0, tool_calls=0, tokens=0, llm_calls=0)
    tool_runs = [r for r in st.get("tool_results", []) if r.get("key") and not r.get("cached")]
    from app.agents.nodes import _sources                 # what the agent could cite: passages, record, tool and rule sources
    available = set(_sources(st)) if st.get("request") else set()
    critic = st.get("critic") or {}
    js = st.get("json_stats") or {}
    return dict(id=q["id"], output=st.get("output"), route=st.get("route"), template=(st.get("plan") or {}).get("template"),
                steps=st.get("steps", 0), llm_calls=st.get("llm_calls", 0), tokens=st.get("tokens", 0),
                tool_calls=len(tool_runs), tool_failures=[f"{r['tool']}: {r['error']}" for r in tool_runs if not r["ok"]],
                latency_ms=(st.get("latency_ms") or 0) + (st.get("cached_latency_ms") or 0),   # cached calls: original latency
                cached_llm_calls=st.get("cached_llm_calls", 0), cached_estimated=bool(st.get("cached_estimated")), json_stats=js, json_defaulted=js.get("defaulted", 0),
                replanned=bool(st.get("replanned")), critic_problems_unresolved=critic.get("problems", []) if not critic.get("ok") else [],
                partial_reasons=st.get("partial_reasons", []), available_docs=sorted(available),
                trace=[{k: v for k, v in t.items() if k in ("node", "route", "template", "kind", "ok", "problems")} for t in st.get("trace", [])],
                trace_id=st.get("trace_id"), token_cap=token_cap(st) if st.get("request") else None,
                dense_fallbacks=len(DENSE_FALLBACKS) - fb0)


def completed(run: dict) -> bool:
    """Task completion (template V): a complete answer, a correct-shape refusal or a clarification, within the limits."""
    o = run.get("output")
    return bool(o) and not run.get("crashed") and o.get("behaviour") != "PARTIAL" and run["steps"] <= MAX_STEPS and \
        (run.get("token_cap") is None or run["tokens"] <= run["token_cap"])


def evaluate(qs: list[dict], runs: list[dict], split_label: str) -> dict:
    preds = {r["id"]: r["output"] for r in runs if r.get("output")}
    summary, graded = grade(preds, split="all", ids={q["id"] for q in qs})
    gq = {g["id"]: g for g in graded}
    rows, tax = [], Counter()
    for q, r in zip(qs, runs):
        g = gq[q["id"]]
        run_for_labels = {"response": r.get("output"), "call_error": r.get("crashed"),
                          "valid_first_try": not (r.get("json_stats") or {}).get("retried") and not r.get("json_defaulted"),
                          "context_docs": r.get("available_docs", [])}
        labels = classify(q, run_for_labels, g)
        cls = primary_class(q, {**r, "route": r.get("route")}, labels) if not g["passed"] else None
        if cls:
            tax[cls] += 1
        rows.append({**{k: r.get(k) for k in ("id", "route", "template", "steps", "llm_calls", "tool_calls", "tokens", "latency_ms",
                                              "replanned", "trace_id")},
                     "category": q["category"], "passed": g["passed"], "completed": completed(r), "labels": labels,
                     "failure_class": cls, "checks": g["checks"], "crashed": r.get("crashed"),
                     "tool_failures": r.get("tool_failures"), "partial_reasons": r.get("partial_reasons"),
                     "critic_problems_unresolved": r.get("critic_problems_unresolved"), "trace": r.get("trace"),
                     "output": r.get("output")})                    # kept for the usefulness judge (4.1)
    n = len(rows)
    js = Counter()
    for r in runs:
        js.update(r.get("json_stats") or {})
    kinds = Counter()
    calls_by_kind: dict[str, list[int]] = {}
    for q, r in zip(qs, runs):
        k = q["request"]["request_kind"] if r.get("route") == "PLAN" else (r.get("route") or "CRASH")
        kinds[k] += 1
        calls_by_kind.setdefault(k, []).append(r.get("tool_calls", 0))
    toks = [r.get("tokens", 0) for r in runs]
    lat = [r.get("latency_ms") or 0 for r in runs]
    metrics = dict(
        split=split_label, queries=n, accuracy=summary["accuracy"],
        task_completion_rate=round(sum(r["completed"] for r in rows) / n, 3) if n else None,
        avg_tool_calls_per_task=round(sum(r.get("tool_calls", 0) for r in runs) / n, 2) if n else None,
        avg_tool_calls_by_kind={k: round(sum(v) / len(v), 2) for k, v in calls_by_kind.items()},
        tokens_median=int(statistics.median(toks)) if toks else 0, tokens_p95=pct(toks, 95), tokens_total=sum(toks),
        latency_p50_ms=pct(lat, 50), latency_p95_ms=pct(lat, 95),
        within_step_cap=round(sum(r.get("steps", 0) <= MAX_STEPS for r in runs) / n, 3) if n else None,
        max_steps=max((r.get("steps", 0) for r in runs), default=0), replans=sum(r["replanned"] for r in rows),
        dense_fallbacks=sum(r.get("dense_fallbacks", 0) for r in runs),
        tool_retries=sum((r.get("json_stats") or {}).get("tool_retries", 0) for r in runs),
        llm_calls_total=sum(r.get("llm_calls", 0) for r in runs), cached_llm_calls=sum(r.get("cached_llm_calls", 0) for r in runs),
        queries_with_estimated_cache=sum(r.get("cached_estimated", False) for r in runs),
        tokens_median_answered=int(statistics.median([t for t in toks if t])) if any(toks) else 0,
        json=dict(js), failure_taxonomy={c: tax.get(c, 0) for c in CLASSES}, by_category=summary["by_category"],
        **{k: summary[k] for k in ("band_accuracy", "recommendation_accuracy", "score_reproducibility", "checklist_accuracy",
                                   "finding_recall", "finding_precision", "fact_recall", "citation_hit_rate", "citation_precision",
                                   "refusal_accuracy", "injection_resistance", "pii_leak_free")})
    if js.get("calls"):
        metrics["json_first_try_rate"] = round(js.get("first_try", 0) / js["calls"], 3)
        metrics["json_defaulted_rate"] = round(js.get("defaulted", 0) / js["calls"], 3)
    return {"metrics": metrics, "rows": rows}


def write_markdown(report: dict, path):
    L = ["# Agent evaluation (Week 3, steps 3.8-3.9)", "",
         f"Model: `{report['model']}` · backend `{report['backend']}` · constrained decoding: `{report['constrained'] or 'off'}` · "
         f"retrieval `{report['retrieval']}`. Generated by `python -m scripts.agent_eval`; notes go under *Team notes*.", ""]
    for split, res in report["splits"].items():
        m = res["metrics"]
        L += [f"## {split} ({m['queries']} queries, run {res['run_at']})", "",
              "### Template section V", "", "| Metric | Value |", "|---|---|",
              f"| End-to-end accuracy | {m['accuracy']} |", f"| Task completion rate | {m['task_completion_rate']} |",
              f"| Average tool calls per task | {m['avg_tool_calls_per_task']} ({', '.join(f'{k}: {v}' for k, v in m['avg_tool_calls_by_kind'].items())}) |",
              f"| Tokens per query (median / p95 / total) | {m['tokens_median']:,} / {m['tokens_p95']:,} / {m['tokens_total']:,} "
              f"(median over queries that called the model: {m.get('tokens_median_answered', 0):,}) |",
              f"| Latency per query (p50 / p95) | {m['latency_p50_ms'] / 1000:.1f} s / {m['latency_p95_ms'] / 1000:.1f} s |",
              f"| Within the 10-step cap | {m['within_step_cap']} (max {m['max_steps']}) |", ""]
        if m.get("cached_llm_calls"):
            L += [f"{m['cached_llm_calls']} of {m['llm_calls_total']} LLM calls were replayed from the response cache; their "
                  "original tokens and latency are counted, so cost and latency read as if uncached"
                  + (f" ({m['queries_with_estimated_cache']} queries used older cache files: tokens estimated at "
                     "characters / 3.6, latency unknown and counted as 0)." if m.get("queries_with_estimated_cache") else "."), ""]
        L += ["### Failure taxonomy (primary class per failed query)", "", "| Class | Queries | Examples |", "|---|---|---|"]
        for c in CLASSES:
            ex = [r for r in res["rows"] if r["failure_class"] == c][:2]
            L.append(f"| {c} | {m['failure_taxonomy'][c]} | " + "; ".join(f"{r['id']} ({', '.join(r['labels'][:3]) or '-'})" for r in ex) + " |")
        L += ["", "### Detailed checks" + (" (dev: against the Week 1 single-prompt baselines)" if split == "dev" else ""), "",
              "| Check | Agent | Week 1 few-shot / CoT-SC / decomposition |", "|---|---|---|"]
        for k in ("accuracy", "band_accuracy", "checklist_accuracy", "finding_recall", "recommendation_accuracy", "fact_recall",
                  "citation_hit_rate", "citation_precision", "refusal_accuracy", "injection_resistance", "pii_leak_free"):
            base = " / ".join(str(x) for x in WEEK1_DEV[k]) if split == "dev" and k in WEEK1_DEV else "–"
            L.append(f"| {k} | {m.get(k)} | {base} |")
        L += [f"| tokens per query (median) | {m['tokens_median']:,} | " + (" / ".join(f"{x:,}" for x in WEEK1_DEV['tokens_per_query']) if split == "dev" else "–") + " |",
              "", f"By category: {', '.join(f'{k} {v}' for k, v in m['by_category'].items())}.", "",
              f"Reliability layer: {json.dumps(m['json'])}; replans: {m['replans']}.", ""]
    L += ["## Reading these numbers", "",
          "- Band, score, checklist, recommendation and approvers come from the rubric engine through the tools (D2, D24): "
          "they measure that the agent passes tool output through unchanged, not model skill.",
          "- The agent's own contribution shows in routing (refuse / clarify / plan), fact recall and citations on questions, "
          "injection reporting, and completing within the step and token budgets.", "", "## Team notes", "",
          "*(M1/M5: add the analysis of the worst failure clusters here; the generator keeps everything below this heading.)*", ""]
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    text = "\n".join(L)
    if "## Team notes" in old:
        text = text[:text.index("## Team notes")] + old[old.index("## Team notes"):]
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    qs = [q for q in load_eval() if q["split"] == a.split]
    if a.ids:
        want = {x.strip() for x in a.ids.split(",")}
        qs = [q for q in load_eval() if q["id"] in want]
    elif a.limit:
        qs = qs[:: max(1, len(qs) // a.limit)][: a.limit]               # spread over categories
    smoke = bool(a.limit or a.ids)
    mock = llm_client.backend() == "mock"
    out_dir = ROOT / "reports" / ("mock" if mock else "smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / ("agent_eval_smoke.json" if smoke else "agent_eval_v1.json")
    warm_up()
    runs = []
    for i, q in enumerate(qs, 1):
        r = run_one(q)
        o = r.get("output") or {}
        print(f"[{i}/{len(qs)}] {q['id']} {q['category']:<28} -> {o.get('behaviour', 'CRASH'):<8} steps={r['steps']:<2} "
              f"tools={r['tool_calls']} tokens={r['tokens']:<6} {(r.get('latency_ms') or 0) / 1000:.1f}s"
              + (f"  {r['crashed']}" if r.get("crashed") else ""), flush=True)
        runs.append(r)
    res = evaluate(qs, runs, a.split if not smoke else f"{a.split} smoke")
    res["run_at"] = time.strftime("%Y-%m-%d %H:%M")
    model = "mock" if mock else os.getenv("LLM_MODEL", "?")
    report = {"model": model, "backend": llm_client.backend(), "constrained": os.getenv("LLM_CONSTRAINED", ""),
              "retrieval": f"{os.getenv('RETRIEVAL_STRATEGY', 'fixed512_ctx')} / {os.getenv('RETRIEVAL_MODE', 'auto')} / "
                           f"rerank {os.getenv('RERANK_BACKEND', 'cross_encoder')}"
                           + ("" if os.getenv("RERANK_BACKEND", "cross_encoder") in ("none", "off", "0") else
                              f" {os.getenv('RERANK_MODEL', 'minilm')}@{os.getenv('RERANK_CANDIDATES', '12')}"), "splits": {}}
    if rep_path.exists() and not smoke:
        try:
            old = json.loads(rep_path.read_text(encoding="utf-8"))
            if old.get("model") == model:
                report["splits"] = old.get("splits", {})
        except ValueError:
            pass
    report["splits"][res["metrics"]["split"]] = res
    rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    doc = (out_dir / "agent_eval.md") if (smoke or mock) else ROOT / "docs" / "agent_eval.md"
    write_markdown(report, doc)
    m = res["metrics"]
    print(f"\naccuracy {m['accuracy']} | completion {m['task_completion_rate']} | tool calls/task {m['avg_tool_calls_per_task']} | "
          f"tokens median {m['tokens_median']:,} (model-calling queries {m['tokens_median_answered']:,}) | latency p50 {(m['latency_p50_ms'] or 0) / 1000:.1f}s p95 {(m['latency_p95_ms'] or 0) / 1000:.1f}s")
    print("failure taxonomy:", m["failure_taxonomy"])
    if m["dense_fallbacks"] or m["tool_retries"]:
        print(f"infrastructure: {m['dense_fallbacks']} search(es) fell back to BM25 (embeddings endpoint slow or down), "
              f"{m['tool_retries']} timed-out tool call(s) retried")
    if m["cached_llm_calls"]:
        print(f"cache: {m['cached_llm_calls']}/{m['llm_calls_total']} LLM calls replayed (original tokens/latency counted"
              + (f"; {m['queries_with_estimated_cache']} queries from old cache files: tokens estimated, latency unknown)"
                 if m["queries_with_estimated_cache"] else ")"))
    print(f"wrote {rep_path.relative_to(ROOT)} and {doc.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
