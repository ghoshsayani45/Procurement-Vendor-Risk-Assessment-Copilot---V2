"""The one retrieval entry point the tools and agents use (document_search wraps this).

    from app.retrieval.service import search
    passages = search("liability cap", vendor_ids=["V009"])      # <= 6 Passages, each with .citation()

Built once per process from the store (STORE_BACKEND), with the chunking strategy in RETRIEVAL_STRATEGY.
RETRIEVAL_MODE: auto (default) = hybrid when every chunk already has a vector for the configured embedding model, else
BM25 only (never starts a long embedding run inside the API); hybrid = always (embeds what is missing); bm25 = never dense.
Run `python -m scripts.index_corpus --strategies <RETRIEVAL_STRATEGY> --embed` first for hybrid.
"""
from __future__ import annotations

import os
import threading

from app.retrieval.chunking import load_corpus, parents_for
from app.retrieval.retrievers import Filters, Passage, RetrievalPipeline

_lock = threading.Lock()
_pipelines: dict[tuple, tuple[RetrievalPipeline, str]] = {}   # (store, strategy, mode, embed, rerank) -> (pipeline, note)
_mode_note = ""


def build_pipeline(store=None) -> tuple[RetrievalPipeline, str]:
    from app.db.store import get_store
    strategy = os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx")
    mode = os.getenv("RETRIEVAL_MODE", "auto").strip().lower()
    store = store or get_store()
    chunks = store.load_chunks(strategy)
    if not chunks:
        raise RuntimeError(f"no '{strategy}' chunks in the {store.backend} store: run `python -m scripts.index_corpus "
                           f"--strategies {strategy}` first")
    parents = parents_for(load_corpus()) if strategy == "small2big" else None
    note, use_dense, embedder = "", mode != "bm25", None
    if use_dense:
        try:
            from app.retrieval.retrievers import embeddings_ready, get_embedder
            embedder = get_embedder()
            if mode == "auto" and not embeddings_ready(chunks, embedder.model_id, store):
                use_dense, note = False, (f"BM25 only: '{strategy}' chunks are not embedded with {embedder.model_id} yet "
                                          f"(run index_corpus --strategies {strategy} --embed)")
        except Exception as e:
            if mode == "hybrid":
                raise
            use_dense, note = False, f"BM25 only: embeddings unavailable ({type(e).__name__}: {str(e)[:120]})"
    try:
        pipe = RetrievalPipeline(chunks, parents=parents, store=store, embedder=embedder, use_dense=use_dense)
    except Exception as e:
        if mode == "hybrid" or not use_dense:
            raise
        pipe = RetrievalPipeline(chunks, parents=parents, store=store, use_dense=False)
        note = f"BM25 only: dense index failed ({type(e).__name__}: {str(e)[:120]})"
    note = "; ".join(x for x in (note, pipe.rerank_note) if x)
    if note:
        print(f"[retrieval] {note}")
    return pipe, note


def _key(store) -> tuple:
    return (id(store) if store is not None else os.getenv("STORE_BACKEND", "sqlite") + os.getenv("STORE_SQLITE_PATH", "")
            + os.getenv("DATABASE_URL", ""), os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx"), os.getenv("RETRIEVAL_MODE", "auto"),
            os.getenv("EMBED_BACKEND", "gemini"), os.getenv("EMBED_REUSE_VECTORS_FROM", ""), os.getenv("RERANK_BACKEND", "cross_encoder"), os.getenv("RERANK_MODEL", "minilm"),
            os.getenv("RERANK_CANDIDATES", "12"))


def get_pipeline(store=None) -> RetrievalPipeline:
    """One pipeline per store and retrieval settings, built on first use."""
    global _mode_note
    key = _key(store)
    with _lock:
        if key not in _pipelines:
            _pipelines[key] = build_pipeline(store)
        pipe, _mode_note = _pipelines[key]
        return pipe


def mode_note() -> str:
    return _mode_note


def reset():
    """Forget the cached pipeline (tests, or after re-indexing)."""
    global _mode_note
    with _lock:
        _pipelines.clear()
        _mode_note = ""


def search(query: str, vendor_ids: list[str] | None = None, include_superseded: bool = True,
           doc_types: list[str] | None = None, access_classes: tuple[str, ...] = ("public", "internal"),
           keep: int = 6, store=None) -> list[Passage]:
    f = Filters(vendor_ids=tuple(vendor_ids) if vendor_ids else None, access_classes=tuple(access_classes),
                include_superseded=include_superseded, doc_types=tuple(doc_types) if doc_types else None)
    return get_pipeline(store).search(query, f, keep)
