"""retrieval measured on its own: BM25 vs dense vs hybrid vs hybrid + reranker (+ vendor filter).

    python -m scripts.retrieval_eval                         # all configurations (Gemini embeddings + local reranker)
    python -m scripts.retrieval_eval --configs bm25,bm25_filter   # offline subset
    RERANK_BACKEND=none python -m scripts.retrieval_eval      # skip the reranker rows
    python -m scripts.retrieval_eval --configs hybrid_rerank,production --splits dev --rerank-candidates 20
                                                             # add reranker rows later; earlier rows are kept

Configurations (chunking strategy from --strategy / RETRIEVAL_STRATEGY):
  bm25            BM25 top 60 chunks -> passages                                     (step 1.8 baseline, new chunking)
  bm25_filter     same, with the case's vendor filter applied before search
  dense           embeddings top 60 -> passages
  hybrid          BM25 top 20 + dense top 20, reciprocal rank fusion, dedupe, <= 3 passages per document
  hybrid_filter   hybrid with the vendor filter, no reranker (the shipped default when the reranker is off, D29)
  hybrid_rerank   hybrid candidates reordered by the cross-encoder
  production      hybrid_rerank with the vendor filter (top 6 go to the generator)
Metrics: hit@k, recall@5/10, MRR (document level) and quote@5/10 (passage level), on dev and test, per category.
Writes reports/retrieval_v2.json and docs/experiments/retrieval.md (the template's retrieval table).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from dotenv import load_dotenv

from app.retrieval.chunking import ROOT, STRATEGIES, chunk_corpus, load_corpus, parents_for
from app.retrieval.evaluation import evaluate, load_queries, query_filters, summarise
from app.retrieval.retrievers import (BM25Retriever, DenseRetriever, HybridRetriever, get_embedder, get_reranker, rerank,
                                      to_passages)

load_dotenv()
REPORT = ROOT / "reports" / "retrieval_v2.json"
DOC = ROOT / "docs" / "experiments" / "retrieval.md"
CONFIGS = ("bm25", "bm25_filter", "dense", "hybrid", "hybrid_filter", "hybrid_rerank", "production")
LABELS = {"bm25": "BM25", "bm25_filter": "BM25 + vendor filter", "dense": "Dense", "hybrid": "Hybrid (RRF)", "hybrid_filter": "Hybrid + vendor filter",
          "hybrid_rerank": "Hybrid + reranker", "production": "Hybrid + reranker + vendor filter"}


def chunks_for(strategy, docs):
    try:
        from app.db.store import get_store
        store = get_store()
        chunks = store.load_chunks(strategy)
        if chunks:
            return chunks, store
    except Exception as e:
        print(f"store unavailable ({type(e).__name__}); chunking in memory")
    return chunk_corpus(docs, strategy), None


def run(configs, strategy, splits=("dev", "test"), rerank_candidates: int = 12):
    docs = load_corpus()
    doc_types = {d["doc_id"]: d["doc_type"] for d in docs}
    parents = parents_for(docs) if strategy == "small2big" else None
    chunks, store = chunks_for(strategy, docs)
    fetch = 200 if strategy == "small2big" else 60
    queries = {s: load_queries((s,)) for s in splits}
    report = dict(run_at=time.strftime("%Y-%m-%d %H:%M"), strategy=strategy, chunks=len(chunks), configs={},
                  embed_backend=os.getenv("EMBED_BACKEND", "gemini"), rerank_backend=os.getenv("RERANK_BACKEND", "cross_encoder"))
    parts: dict = {}

    def need(part):
        if part not in parts:
            try:
                if part == "bm25":
                    parts[part] = BM25Retriever(chunks)
                elif part == "dense":
                    emb = get_embedder()
                    report["embed_model"] = emb.model_id
                    parts[part] = DenseRetriever(chunks, emb, store)
                elif part == "reranker":
                    rr = get_reranker()
                    if rr is None:
                        raise RuntimeError("RERANK_BACKEND=none")
                    report["rerank_model"] = rr.model_name
                    parts[part] = rr
            except Exception as e:
                parts[part] = e
        if isinstance(parts[part], Exception):
            raise parts[part]
        return parts[part]

    for cfg in configs:
        try:
            use_filter = cfg in ("bm25_filter", "hybrid_filter", "production")
            if cfg in ("bm25", "bm25_filter"):
                r = need("bm25")
                search = lambda q, r=r, f=use_filter: to_passages(r.search(q["request"]["question"], fetch, query_filters(q, f)), parents)
            elif cfg == "dense":
                r = need("dense")
                search = lambda q, r=r: to_passages(r.search(q["request"]["question"], fetch), parents)
            else:
                h = HybridRetriever(need("bm25"), need("dense"), n_each=20 if strategy != "small2big" else 60)
                rr = need("reranker") if cfg in ("hybrid_rerank", "production") else None

                def search(q, h=h, rr=rr, f=use_filter):
                    cands = to_passages(h.search(q["request"]["question"], 2 * h.n_each, query_filters(q, f)), parents, max_per_doc=3)
                    if rr is None:
                        return cands
                    # rerank only the top N fused candidates (CPU cost); the rest keep their fused order below them
                    return rerank(q["request"]["question"], cands, rr, candidates=rerank_candidates)
        except Exception as e:
            report["configs"][cfg] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{cfg:14s} SKIPPED: {report['configs'][cfg]['error']}")
            continue
        res = {}
        for split, qs in queries.items():
            rows = evaluate(search, qs, doc_types)
            res[split] = {"summary": summarise(rows, "category"), "by_group": summarise(rows, "group"), "per_query": rows}
            a = res[split]["summary"]["ALL"]
            print(f"{cfg:14s} {split:4s} n={a['n']:2d} hit@5={a['hit@5']:.3f} recall@5={a['recall@5']:.3f} recall@10={a['recall@10']:.3f} "
                  f"mrr={a['mrr']:.3f} quote@5={a.get('quote@5', float('nan')):.3f} {a['latency_ms']:.0f}ms", flush=True)
        report["configs"][cfg] = res
    report["rerank_candidates"] = rerank_candidates
    return report


def merge_previous(report: dict) -> dict:
    """Keep configurations from the last run (same strategy) that this run did not redo, so the table can be built in
    parts: e.g. everything without the reranker first, then the reranker rows on a slow CPU."""
    if not REPORT.exists():
        return report
    try:
        old = json.loads(REPORT.read_text(encoding="utf-8"))
    except ValueError:
        return report
    if old.get("strategy") != report["strategy"]:
        return report
    for cfg, res in old.get("configs", {}).items():
        mine = report["configs"].get(cfg)
        if mine is None or ("error" in mine and "error" not in res):
            report["configs"][cfg] = res
    for k in ("embed_model", "rerank_model"):
        report.setdefault(k, old.get(k))
    report["configs"] = {c: report["configs"][c] for c in CONFIGS if c in report["configs"]}
    return report


def f2(v):
    return "–" if v is None else f"{v:.2f}"


def write_markdown(report):
    L = ["# Step 2.4: Retrieval evaluation (BM25, dense, hybrid, reranker)", "",
         f"Run {report['run_at']}. Chunking: `{report['strategy']}` ({report['chunks']:,} chunks). Embeddings: "
         f"`{report.get('embed_model', report['embed_backend'])}`. Reranker: `{report.get('rerank_model', report['rerank_backend'])}`. "
         "Query = the user's raw question. Document-level metrics as in step 1.8; quote@k = share of gold quotes inside a top-k passage. "
         "Generated by `python -m scripts.retrieval_eval`; notes go under *Team notes*.", ""]
    for split in ("dev", "test"):
        L += [f"## {split} split", "", "| Configuration | n | hit@5 | recall@5 | recall@10 | MRR | quote@5 | quote@10 | median latency |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cfg, res in report["configs"].items():
            if "error" in res:
                L.append(f"| {LABELS[cfg]} | – | not run: {res['error'][:90]} | | | | | | |")
                continue
            if split not in res:
                L.append(f"| {LABELS[cfg]} | – | not run on {split} | | | | | | |")
                continue
            a = res[split]["summary"]["ALL"]
            L.append(f"| {LABELS[cfg]} | {a['n']} | {f2(a['hit@5'])} | {f2(a['recall@5'])} | {f2(a['recall@10'])} | {f2(a['mrr'])} | "
                     f"{f2(a.get('quote@5'))} | {f2(a.get('quote@10'))} | {a['latency_ms']:.0f} ms |")
        L.append("")
        ok = [(c, r) for c, r in report["configs"].items() if "error" not in r and split in r]
        if ok:
            cats = sorted({c for _, r in ok for c in r[split]["summary"] if c != "ALL"})
            L += [f"Recall@10 by category ({split}):", "", "| Category | n | " + " | ".join(LABELS[c] for c, _ in ok) + " |",
                  "|---|---|" + "---|" * len(ok)]
            for cat in cats:
                n = ok[0][1][split]["summary"].get(cat, {}).get("n", 0)
                L.append(f"| {cat} | {n} | " + " | ".join(f2(r[split]["summary"].get(cat, {}).get("recall@10")) for _, r in ok) + " |")
            L.append("")
    L += ["## Reading the table", "",
          "- Each row adds one component, so the gain of each is the difference from the row above (filter, dense, fusion, reranker).",
          "- Full reviews have ~10 gold documents, so their recall@10 is capped by construction; the Week 3 agent retrieves per checklist item.",
          "- `production` is what `document_search` runs; only its top 6 passages reach the generator, so quote@5 is the number that matters most for answers.",
          "- Test is reported for transparency; choices are made on dev (D12).", "", "## Team notes", "",
          "*(M2/M1: add observations here; the generator keeps everything below this heading.)*", ""]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    text = "\n".join(L)
    if "## Team notes" in old:
        text = text[:text.index("## Team notes")] + old[old.index("## Team notes"):]
    DOC.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--strategy", default=os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx"), choices=STRATEGIES)
    ap.add_argument("--splits", default="dev,test", help="dev,test (default) or dev only, e.g. for slow reranker runs")
    ap.add_argument("--rerank-candidates", type=int, default=int(os.getenv("RERANK_CANDIDATES", "12")),
                    help="rerank only the top N fused candidates (CPU cost; production default 12, decision 9)")
    ap.add_argument("--fresh", action="store_true", help="do not keep configurations from the previous run")
    a = ap.parse_args(argv)
    report = run([c.strip() for c in a.configs.split(",")], a.strategy, tuple(x.strip() for x in a.splits.split(",") if x.strip()),
                 a.rerank_candidates)
    if not a.fresh:
        report = merge_previous(report)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    write_markdown(report)
    print(f"wrote {REPORT.relative_to(ROOT)} and {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
