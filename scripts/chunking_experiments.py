"""Step 2.2: chunking experiments. Compare chunking strategies on retrieval alone (no generator).

    python -m scripts.chunking_experiments                          # BM25 only: runs anywhere, no model download
    python -m scripts.chunking_experiments --retrievers bm25,dense,hybrid --strategies fixed512_ctx,structure
                                                         # dense needs embeddings: run index_corpus --embed for those strategies first

Strategies (app/retrieval/chunking.py): fixed512 (Week 1 baseline), fixed512_ctx (same + contextual header),
structure (heading/clause-aware sections <= ~400 tokens + header), small2big (128-token children -> parent section).
Everything else is held fixed: raw question as the query, no metadata filter, no reranker.

Selection is made on the dev split (decision D12 keeps test for gates); test is reported read-only as a confirmation.
Rule, per document group: highest quote@5 (does the passage the generator reads contain the evidence?), then recall@10,
then MRR; ties go to the strategy with fewer context words. Groups with fewer than 3 dev queries take the overall winner.
Writes reports/chunking_v1.json and docs/experiments/chunking.md.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

from dotenv import load_dotenv

from app.retrieval.chunking import ROOT, STRATEGIES, chunk_corpus, load_corpus, parents_for
from app.retrieval.evaluation import evaluate, load_queries, summarise
from app.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, to_passages

load_dotenv()
REPORT = ROOT / "reports" / "chunking_v1.json"
DOC = ROOT / "docs" / "experiments" / "chunking.md"
MIN_DEV_QUERIES = 3


def load_chunks(strategy, docs, use_store=True):
    if use_store:
        try:
            from app.db.store import get_store
            store = get_store()
            chunks = store.load_chunks(strategy)
            if chunks:
                return chunks, store
        except Exception as e:   # store not built / Postgres not running: fall back to chunking in memory
            print(f"  store unavailable ({type(e).__name__}); chunking in memory")
    return chunk_corpus(docs, strategy), None


def build(name, chunks, store, cache):
    if name == "bm25":
        return cache.setdefault("bm25", BM25Retriever(chunks))
    if name == "dense":
        return cache.setdefault("dense", DenseRetriever(chunks, store=store))
    if name == "hybrid":
        return HybridRetriever(build("bm25", chunks, store, cache), build("dense", chunks, store, cache))
    raise ValueError(name)


def pick_winner(summ_by_strategy: dict[str, dict], group: str) -> tuple[str, str]:
    def key(s):
        m = summ_by_strategy[s].get(group, {})
        return (m.get("quote@5", -1), m.get("recall@10", 0), m.get("mrr", 0), -m.get("ctx_words@6", 1e9))
    cands = [s for s in summ_by_strategy if group in summ_by_strategy[s]]
    best = max(cands, key=key)
    m = summ_by_strategy[best][group]
    basis = "quote@5" if "quote@5" in m else "recall@10"
    return best, basis


def run(retrievers, strategies, splits=("dev", "test"), use_store=True):
    docs = load_corpus()
    doc_types = {d["doc_id"]: d["doc_type"] for d in docs}
    parents = parents_for(docs)
    queries = {s: load_queries((s,)) for s in splits}
    report = dict(run_at=time.strftime("%Y-%m-%d %H:%M"), documents=len(docs), strategies={}, retrievers=retrievers)
    for strategy in strategies:
        chunks, store = load_chunks(strategy, docs, use_store)
        toks = [c.tokens for c in chunks]
        entry = dict(chunks=len(chunks), median_tokens=st.median(toks), p95_tokens=sorted(toks)[int(0.95 * len(toks))], results={})
        cache = {}
        for name in retrievers:
            try:
                r = build(name, chunks, store, cache)
            except Exception as e:
                entry["results"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
                print(f"{strategy:13s} {name:6s} SKIPPED: {entry['results'][name]['error']}")
                continue
            k = 200 if strategy == "small2big" else 60   # children collapse into parents: fetch more to fill 20 passages
            search = lambda q, r=r: to_passages(r.search(q["request"]["question"], k), parents if strategy == "small2big" else None)
            res = {}
            for split, qs in queries.items():
                rows = evaluate(search, qs, doc_types)
                res[split] = {"by_group": summarise(rows, "group"), "by_category": summarise(rows, "category"), "per_query": rows}
                a = res[split]["by_group"]["ALL"]
                print(f"{strategy:13s} {name:6s} {split:4s} n={a['n']:2d} hit@5={a['hit@5']:.3f} recall@10={a['recall@10']:.3f} "
                      f"mrr={a['mrr']:.3f} quote@5={a.get('quote@5', float('nan')):.3f} quote@10={a.get('quote@10', float('nan')):.3f} "
                      f"ctx={a['ctx_words@6']:.0f}w", flush=True)
            entry["results"][name] = res
        report["strategies"][strategy] = entry

    # winners: per retriever, on dev, per document group
    report["winners"] = {}
    for name in retrievers:
        ok = {s: e["results"][name]["dev"]["by_group"] for s, e in report["strategies"].items()
              if name in e["results"] and "dev" in e["results"][name]}
        if not ok:
            continue
        overall, basis = pick_winner(ok, "ALL")
        w = {"ALL": {"strategy": overall, "basis": basis}}
        for g in sorted({g for summ in ok.values() for g in summ if g != "ALL"}):
            n = next(iter(ok.values())).get(g, {}).get("n", 0)
            if n < MIN_DEV_QUERIES:
                w[g] = {"strategy": overall, "basis": f"only {n} dev queries: overall winner"}
            else:
                s, b = pick_winner(ok, g)
                w[g] = {"strategy": s, "basis": b}
        report["winners"][name] = w
    return report


def fmt(v, digits=2):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def write_markdown(report):
    L = ["# Step 2.2: Chunking experiments", "",
         f"Run {report['run_at']} on corpus v1 ({report['documents']} documents). Query = the user's raw question; no metadata "
         "filter; no reranker. Document-level metrics as in step 1.8, plus **quote@k**: the share of gold quotes found inside a "
         "top-k passage, i.e. whether the text the generator would read actually contains the evidence. Generated by "
         "`python -m scripts.chunking_experiments`; do not edit by hand (add notes under *Team notes* at the end).", "",
         "## Strategies", "", "| Strategy | What it does | Chunks | Median tokens | p95 tokens |", "|---|---|---|---|---|"]
    what = {"fixed512": "Week 1 baseline: 512-token word windows, 64 overlap, no structure",
            "fixed512_ctx": "Same windows + contextual header `[title | vendor]` (isolates the header)",
            "structure": "Split at headings/clauses (markdown, 4.1, Rule/Article/Chapter, ALL CAPS, control ids); whole sections packed to ≤ ~400 tokens; header `[title | vendor | section]`",
            "small2big": "128-token children are searched; each hit returns its parent `structure` section"}
    for s, e in report["strategies"].items():
        L.append(f"| `{s}` | {what.get(s, '')} | {e['chunks']:,} | {e['median_tokens']:.0f} | {e['p95_tokens']} |")
    L.append("")
    for split in ("dev", "test"):
        for name in report["retrievers"]:
            rows = [(s, e["results"].get(name, {})) for s, e in report["strategies"].items()]
            rows = [(s, r[split]["by_group"]["ALL"]) for s, r in rows if split in r]
            if not rows:
                continue
            title = "selection" if split == "dev" else "confirmation only, not used to choose"
            L += [f"## {name}: {split} split ({title})", "",
                  "| Strategy | n | hit@5 | recall@5 | recall@10 | MRR | quote@5 | quote@10 | context words in top 6 |", "|---|---|---|---|---|---|---|---|---|"]
            for s, a in rows:
                L.append(f"| `{s}` | {a['n']} | {fmt(a['hit@5'])} | {fmt(a['recall@5'])} | {fmt(a['recall@10'])} | {fmt(a['mrr'])} | "
                         f"{fmt(a.get('quote@5'))} | {fmt(a.get('quote@10'))} | {a['ctx_words@6']:.0f} |")
            L.append("")
            groups = sorted({g for _, r in [(s, e["results"].get(name, {})) for s, e in report["strategies"].items()] if split in r
                             for g in r[split]["by_group"] if g != "ALL"})
            L += ["By document group (quote@5 / recall@10; n = queries):", "",
                  "| Group | n | " + " | ".join(f"`{s}`" for s, _ in rows) + " |", "|---|---|" + "---|" * len(rows)]
            for g in groups:
                cells, n = [], 0
                for s, _ in rows:
                    m = report["strategies"][s]["results"][name][split]["by_group"].get(g)
                    if m:
                        n = m["n"]
                        cells.append(f"{fmt(m.get('quote@5'))} / {fmt(m['recall@10'])}")
                    else:
                        cells.append("–")
                L.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
            L.append("")
    if report.get("winners"):
        L += ["## Choice per document group (dev split)", "", "| Retriever | Group | Chosen strategy | Basis |", "|---|---|---|---|"]
        for name, w in report["winners"].items():
            for g, v in w.items():
                L.append(f"| {name} | {'**ALL**' if g == 'ALL' else g} | `{v['strategy']}` | {v['basis']} |")
        L += ["", "The production strategy is set with `RETRIEVAL_STRATEGY` in `.env` and recorded as a decision in `docs/decisions.md`. "
              "Prefer the choice made with the retriever that ships (hybrid) once dense results exist.", ""]
    L += ["## Caveats", "",
          "- Small samples: dev has 26 queries with gold documents (13 with gold quotes); one query moves a mean by 4–8 points.",
          "- Synthetic vendor documents are short (median ~150 words), so for them most strategies return the whole document; "
          "differences show up mainly on long contracts and policy manuals.",
          "- quote@k needs the quote verbatim inside one passage: a quote split across two fixed windows counts as missed, which is the point.",
          "", "## Team notes", "", "*(M2: add observations here; the generator keeps everything below this heading.)*", ""]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    notes = old[old.index("## Team notes"):] if "## Team notes" in old else None
    text = "\n".join(L)
    if notes:
        text = text[:text.index("## Team notes")] + notes
    DOC.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrievers", default="bm25", help="comma-separated: bm25, dense, hybrid")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--no-store", action="store_true", help="chunk in memory instead of reading the store")
    a = ap.parse_args(argv)
    report = run(a.retrievers.split(","), a.strategies.split(","), use_store=not a.no_store)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    write_markdown(report)
    for name, w in report.get("winners", {}).items():
        print(f"winner ({name}): " + ", ".join(f"{g}={v['strategy']}" for g, v in w.items()))
    print(f"wrote {REPORT.relative_to(ROOT)} and {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
