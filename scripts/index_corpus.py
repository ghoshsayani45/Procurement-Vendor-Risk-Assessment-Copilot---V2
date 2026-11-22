"""load corpus v1 into the document/chunk store (idempotent, versioned).

    python -m scripts.index_corpus                                   # sqlite store, strategy from RETRIEVAL_STRATEGY (structure)
    python -m scripts.index_corpus --strategies fixed512,fixed512_ctx,structure,small2big
    python -m scripts.index_corpus --embed                           # also compute missing embeddings (EMBED_BACKEND, gemini)
    python -m scripts.index_corpus --strategies structure --embed    # Gemini free tier: one strategy is ~2 h of pacing
    STORE_BACKEND=postgres python -m scripts.index_corpus --embed     # Postgres + pgvector from `make up`

Steps:
  1. read data/processed/corpus_v1.jsonl and check it against the frozen manifest (doc ids and SHA-1 of every text);
  2. normalise metadata: vendor ids (V000 form or null), dates (ISO yyyy-mm-dd or null; the raw value is kept as
     `version`), access class and status enums;
  3. upsert vendors (data/seed/vendors.csv), vendor records (vendors.json minus answer-key fields, D24) and documents;
     documents whose SHA-1 is unchanged are skipped;
  4. per strategy, re-chunk only new/changed documents or all documents after a CHUNKER_VERSION bump;
  5. with --embed, embed only chunks that have no vector yet for the current model;
  6. record the run in ingest_runs and print what changed. A second run on the same corpus changes nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

from dotenv import load_dotenv

from app.retrieval.chunking import CHUNKER_VERSION, ROOT, STRATEGIES, chunk_corpus, load_corpus

load_dotenv()
CORPUS_VERSION = "v1"
MANIFEST = ROOT / "data" / "manifests" / "corpus_v1_manifest.csv"
VENDORS = ROOT / "data" / "seed" / "vendors.csv"
MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


class IngestError(RuntimeError):
    pass


def norm_date(raw) -> str | None:
    """'2026-01-31' -> same; '2022-06' -> 2022-06-01; '01-Aug-2023' -> 2023-08-01; 'v1 (2021)' -> 2021-01-01; else None."""
    if not raw:
        return None
    s = str(raw)
    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s):
        return m.group(0)
    if m := re.search(r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\b", s):
        if m.group(2).lower() in MONTHS:
            return f"{m.group(3)}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    if m := re.fullmatch(r"(\d{4})-(\d{2})", s.strip()):
        return f"{m.group(1)}-{m.group(2)}-01"
    if m := re.search(r"\b(19|20)\d{2}\b", s):
        return f"{m.group(0)}-01-01"
    return None


def normalise(doc: dict) -> dict:
    d = dict(doc)
    vid = (d.get("vendor_id") or "").strip().upper()
    if vid and not re.fullmatch(r"V\d{3}", vid):
        raise IngestError(f"{d['doc_id']}: bad vendor_id {d.get('vendor_id')!r}")
    d["vendor_id"] = vid or None
    d["version"] = d.get("version") or (str(d["document_date"]) if d.get("document_date") else None)
    d["document_date"] = norm_date(d.get("document_date"))
    d["access_class"] = (d.get("access_class") or "internal").lower()
    d["status"] = (d.get("status") or "CURRENT").upper()
    if d["access_class"] not in ("public", "internal", "restricted"):
        raise IngestError(f"{d['doc_id']}: bad access_class {d['access_class']!r}")
    if d["status"] not in ("CURRENT", "SUPERSEDED"):
        raise IngestError(f"{d['doc_id']}: bad status {d['status']!r}")
    return d


def check_against_manifest(docs: list[dict]) -> None:
    """The corpus file must be exactly the frozen corpus v1: same ids, and each text's SHA-1 equal to the manifest's."""
    man = {r["doc_id"]: r["sha1"] for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))}
    ids = [d["doc_id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise IngestError("duplicate doc_id in corpus_v1.jsonl")
    missing, extra = set(man) - set(ids), set(ids) - set(man)
    bad = [d["doc_id"] for d in docs if d["doc_id"] in man and hashlib.sha1(d["text"].encode()).hexdigest() != man[d["doc_id"]]]
    if missing or extra or bad:
        raise IngestError(f"corpus_v1.jsonl does not match the frozen manifest: {len(missing)} missing, {len(extra)} extra, "
                          f"{len(bad)} changed (e.g. {(sorted(missing) + sorted(extra) + bad)[:3]}). Rebuild with `make corpus`.")


def run(store, strategies: list[str], embed: bool = False, verify: bool = True, docs: list[dict] | None = None) -> dict:
    t0 = time.time()
    docs = docs if docs is not None else load_corpus()
    if verify:
        check_against_manifest(docs)
    docs = [normalise(d) for d in docs]
    report = {"corpus_version": CORPUS_VERSION, "chunker_version": CHUNKER_VERSION, "backend": store.backend}

    vendors = list(csv.DictReader(open(VENDORS, encoding="utf-8")))
    store.upsert_vendors(vendors)
    from app.tools.records import load_records_file
    report["vendor_records_changed"] = store.upsert_vendor_records(load_records_file())
    old = store.document_hashes()
    changed = [d for d in docs if old.get(d["doc_id"]) != d["sha1"]]
    removed = sorted(set(old) - {d["doc_id"] for d in docs})
    if removed:
        store.delete_documents(removed)
    if changed:
        store.upsert_documents(changed, CORPUS_VERSION)
    report["documents"] = {"total": len(docs), "new_or_changed": len(changed), "removed": len(removed),
                           "unchanged": len(docs) - len(changed)}
    changed_ids = {d["doc_id"] for d in changed}

    report["strategies"] = {}
    for strategy in strategies:
        versions = store.chunk_versions(strategy)
        todo = [d for d in docs if d["doc_id"] in changed_ids or versions.get(d["doc_id"]) != CHUNKER_VERSION]
        chunks = chunk_corpus(todo, strategy) if todo else []
        if todo:
            store.replace_chunks(strategy, [d["doc_id"] for d in todo], chunks, CHUNKER_VERSION)
        entry = {"documents_rechunked": len(todo), "chunks_written": len(chunks)}
        if embed:
            from app.retrieval.retrievers import embed_chunks, get_embedder
            emb = get_embedder()
            all_chunks = store.load_chunks(strategy)
            before = len(store.load_embeddings(emb.model_id, [c.chunk_id for c in all_chunks]))
            embed_chunks(all_chunks, emb, store)
            entry.update(embed_model=emb.model_id, embeddings_added=len(all_chunks) - before)
        report["strategies"][strategy] = entry
        store.record_run(CORPUS_VERSION, CHUNKER_VERSION, strategy, {**report["documents"], **entry})
    report["counts"] = store.counts()
    report["seconds"] = round(time.time() - t0, 1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", default=os.getenv("RETRIEVAL_STRATEGY", "structure"),
                    help=f"comma-separated, from {', '.join(STRATEGIES)}")
    ap.add_argument("--embed", action="store_true", help="compute missing embeddings with EMBED_BACKEND")
    ap.add_argument("--backend", default=None, help="sqlite (default) or postgres; overrides STORE_BACKEND")
    a = ap.parse_args(argv)
    from app.db.store import get_store
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGIES:
            sys.exit(f"unknown strategy {s!r}; choose from {', '.join(STRATEGIES)}")
    store = get_store(a.backend)
    try:
        report = run(store, strategies, embed=a.embed)
    except IngestError as e:
        sys.exit(f"INGEST FAILED: {e}")
    except Exception as e:
        if type(e).__name__ == "GeminiQuotaExhausted":     # resumable: everything embedded so far is in the store
            sys.exit(f"STOPPED: {e}")
        raise
    finally:
        store.close()
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
