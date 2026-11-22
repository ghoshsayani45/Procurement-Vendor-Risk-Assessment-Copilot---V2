"""does a cross-encoder reranker earn its CPU cost? Dev split only (D12).

    python -m scripts.rerank_experiment                                  # none, minilm@12, minilm@20, bge-base@12
    python -m scripts.rerank_experiment --variants none,minilm@12         # a subset
    python -m scripts.rerank_experiment --bm25-only                      # without embeddings (offline check only)

Each variant runs the production retrieval path (hybrid BM25 + dense with RRF, vendor filter, <= 3 passages per
document) and then reranks the top N fused candidates with the given model. Query = the user's raw question.
Measured on the dev queries that have gold documents: hit@5, recall@5, MRR, quote@5 (share of gold quotes inside a
top-5 passage: what reaches the generator) and per-query latency (median and p95, first query excluded: model load).

Adoption rule (written before running, so the result cannot move it):
  adopt the cheapest variant with quote@5 >= baseline + 0.03, hit@5 and MRR not lower than the baseline, and p95
  latency <= baseline p95 + 1,500 ms per search (the agent runs at most 3 searches per question);
  otherwise keep the reranker off and record why.
Writes reports/rerank_v1.json and docs/experiments/reranker.md (Team notes kept).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from dotenv import load_dotenv

from app.retrieval.chunking import ROOT, load_corpus
from app.retrieval.evaluation import evaluate, load_queries, query_filters, summarise
from app.retrieval.retrievers import (BM25Retriever, DenseRetriever, HybridRetriever, Reranker, get_embedder, rerank,
                                      rerank_model_name, to_passages)
from scripts.retrieval_eval import chunks_for

load_dotenv()
REPORT = ROOT / "reports" / "rerank_v1.json"
DOC = ROOT / "docs" / "experiments" / "reranker.md"
DEFAULT_VARIANTS = "none,minilm@12,minilm@20,bge-base@12"
QUOTE_GAIN, LATENCY_BUDGET_MS = 0.03, 1500


def parse(v: str) -> tuple[str | None, int]:
    if v == "none":
        return None, 0
    model, _, n = v.partition("@")
    return model, int(n or 12)


def p95(vals):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] if s else 0.0


def run(variants: list[str], strategy: str, bm25_only: bool) -> dict:
    docs = load_corpus()
    doc_types = {d["doc_id"]: d["doc_type"] for d in docs}
    chunks, store = chunks_for(strategy, docs)
    bm25 = BM25Retriever(chunks)
    if bm25_only:
        base = lambda q: to_passages(bm25.search(q["request"]["question"], 40, query_filters(q, True)), None, max_per_doc=3)
        retrieval = "bm25 (offline check: not the production path)"
    else:
        emb = get_embedder()
        hybrid = HybridRetriever(bm25, DenseRetriever(chunks, emb, store), n_each=20)
        base = lambda q: to_passages(hybrid.search(q["request"]["question"], 40, query_filters(q, True)), None, max_per_doc=3)
        retrieval = f"hybrid (bm25 + {emb.model_id}, RRF) + vendor filter"
    qs = load_queries(("dev",))
    report = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "strategy": strategy, "retrieval": retrieval, "queries": len(qs),
              "rule": {"quote@5_gain": QUOTE_GAIN, "p95_latency_budget_ms": LATENCY_BUDGET_MS}, "variants": {}}
    models: dict[str, Reranker] = {}
    for v in variants:
        model, n = parse(v)
        try:
            if model:
                if model not in models:
                    t0 = time.time()
                    models[model] = Reranker(model)
                    print(f"loaded {rerank_model_name(model)} in {time.time() - t0:.1f}s", flush=True)
                rr = models[model]
                search = lambda q, rr=rr, n=n: rerank(q["request"]["question"], base(q), rr, candidates=n)
            else:
                search = base
            search(qs[0])                                            # warm-up: model load and first-call cost excluded
            rows = evaluate(search, qs, doc_types)
        except Exception as e:
            report["variants"][v] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{v:14s} SKIPPED: {report['variants'][v]['error']}", flush=True)
            continue
        a = summarise(rows)["ALL"]
        lat = [r["latency_ms"] for r in rows]
        res = {k: a.get(k) for k in ("n", "hit@5", "recall@5", "recall@10", "mrr", "quote@5", "quote@10")}
        res.update(latency_median_ms=round(sorted(lat)[len(lat) // 2], 1), latency_p95_ms=round(p95(lat), 1),
                   model=rerank_model_name(model) if model else None, candidates=n,
                   by_category={c: m.get("quote@5") for c, m in summarise(rows, "category").items() if c != "ALL"},
                   per_query=[{k: r.get(k) for k in ("id", "hit@5", "mrr", "quote@5", "latency_ms")} for r in rows])
        report["variants"][v] = res
        print(f"{v:14s} hit@5={res['hit@5']:.3f} recall@5={res['recall@5']:.3f} mrr={res['mrr']:.3f} "
              f"quote@5={(res['quote@5'] or 0):.3f} latency median {res['latency_median_ms']:.0f}ms p95 {res['latency_p95_ms']:.0f}ms",
              flush=True)
    report["verdict"] = verdict(report)
    return report


def verdict(report: dict) -> dict:
    base = report["variants"].get("none")
    if not base or "error" in base:
        return {"adopt": None, "reason": "no baseline (variant 'none') to compare with"}
    ran = [v for v, r in report["variants"].items() if v != "none" and "error" not in r]
    if not ran:
        return {"adopt": None, "reason": "no reranker variant ran (see the errors): nothing was measured, so no decision yet"}
    ok = []
    for v, r in report["variants"].items():
        if v == "none" or "error" in r:
            continue
        checks = {"quote@5": (r["quote@5"] or 0) >= (base["quote@5"] or 0) + QUOTE_GAIN,
                  "hit@5": r["hit@5"] >= base["hit@5"], "mrr": r["mrr"] >= base["mrr"],
                  "latency": r["latency_p95_ms"] <= base["latency_p95_ms"] + LATENCY_BUDGET_MS}
        r["checks"] = checks
        if all(checks.values()):
            ok.append((r["latency_p95_ms"], v))
    if ok:
        best = min(ok)[1]
        m, n = parse(best)
        return {"adopt": best, "env": {"RERANK_BACKEND": "cross_encoder", "RERANK_MODEL": m, "RERANK_CANDIDATES": str(n)},
                "reason": f"{best} meets every check of the adoption rule and is the cheapest variant that does"}
    return {"adopt": None, "env": {"RERANK_BACKEND": "none"},
            "reason": "no variant meets the adoption rule: keep the reranker off (see the failed checks per variant)"}


def write_markdown(report: dict):
    v = report["verdict"]
    L = ["# Decision 9: reranker experiment (dev split)", "",
         f"Run {report['run_at']}. Chunking `{report['strategy']}`; retrieval: {report['retrieval']}. {report['queries']} dev "
         "queries with gold documents; query = the user's raw question. Generated by `python -m scripts.rerank_experiment`.", "",
         f"**Adoption rule** (fixed before the run): quote@5 at least +{QUOTE_GAIN:.2f} over no reranker, hit@5 and MRR not "
         f"lower, p95 latency at most +{LATENCY_BUDGET_MS:,} ms per search. Cheapest passing variant wins.", "",
         "| Variant | Model | Candidates | hit@5 | recall@5 | MRR | quote@5 | quote@10 | latency median / p95 | Passes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["variants"].items():
        if "error" in r:
            L.append(f"| {name} | – | – | not run: {r['error'][:80]} | | | | | | |")
            continue
        passes = "baseline" if name == "none" else ("yes" if all(r.get("checks", {}).values()) else
                                                    "no (" + ", ".join(k for k, ok in r.get("checks", {}).items() if not ok) + ")")
        L.append(f"| {name} | {r['model'] or '–'} | {r['candidates'] or '–'} | {r['hit@5']:.2f} | {r['recall@5']:.2f} | "
                 f"{r['mrr']:.2f} | {(r['quote@5'] or 0):.2f} | {(r['quote@10'] or 0):.2f} | "
                 f"{r['latency_median_ms']:.0f} / {r['latency_p95_ms']:.0f} ms | {passes} |")
    L += ["", f"**Verdict:** {'adopt `' + v['adopt'] + '`' if v.get('adopt') else 'keep the reranker off'}. {v['reason']}.", ""]
    if v.get("env"):
        L += ["Settings: " + ", ".join(f"`{k}={val}`" for k, val in v["env"].items()) + ".", ""]
    L += ["## Next", "",
          "- Confirm end to end: `python -m scripts.agent_eval --split dev` with these settings, against the 0.933 dev run "
          "without a reranker (accuracy, citation hit rate, latency p95).",
          "- Record the result as the reranker decision in `docs/decisions.md` (amending D29).", "", "## Team notes", "",
          "*(M2: observations here; the generator keeps everything below this heading.)*", ""]
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    text = "\n".join(L)
    if "## Team notes" in old:
        text = text[:text.index("## Team notes")] + old[old.index("## Team notes"):]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--strategy", default=os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx"))
    ap.add_argument("--bm25-only", action="store_true")
    a = ap.parse_args(argv)
    report = run([v.strip() for v in a.variants.split(",") if v.strip()], a.strategy, a.bm25_only)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    write_markdown(report)
    print(f"\nverdict: {report['verdict']['reason']}")
    print(f"wrote {REPORT.relative_to(ROOT)} and {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
