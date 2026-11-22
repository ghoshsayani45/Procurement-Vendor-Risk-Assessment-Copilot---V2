"""Retrievers: BM25 (sparse), dense (embeddings), hybrid (reciprocal rank fusion) and a cross-encoder reranker.

Every retriever searches *index chunks* and returns ranked (chunk, score) pairs. Metadata filters (vendor, access class,
status, slice) are applied BEFORE ranking, as a mask over the index, so a filtered-out chunk can never be returned.
`RetrievalPipeline.search()` is what the tools call: hybrid -> dedupe -> rerank -> top 6 passages with citation metadata.

Embedding backends (EMBED_BACKEND):
  gemini    Google Gemini embeddings API (default; GEMINI_API_KEY, GEMINI_EMBED_MODEL=gemini-embedding-2, 768 dims),
            paced to the free-tier limits and resumable: see app/retrieval/gemini_embed.py
  local     sentence-transformers on CPU (EMBED_LOCAL_MODEL, default BAAI/bge-small-en-v1.5)
  endpoint  any OpenAI-compatible embeddings endpoint via app.llm_client.embed (EMBED_BASE_URL, EMBED_MODEL, EMBED_API_KEY,
            EMBED_CA_BUNDLE). bge models get the bge query prefix, like `local`; vectors are L2-normalised. If the endpoint
            serves the same model as an earlier backend, EMBED_REUSE_VECTORS_FROM=<that model id> (e.g.
            local_BAAI/bge-small-en-v1.5) reuses the stored vectors: check first with scripts/check_embed_endpoint.py
  hash      deterministic hashed bag-of-words vectors. NOT semantic: only for offline tests and CI.
Reranker backends (RERANK_BACKEND): cross_encoder (default) or none. RERANK_MODEL: minilm (default,
cross-encoder/ms-marco-MiniLM-L-6-v2, 22M parameters), bge-base (BAAI/bge-reranker-base, 278M) or any Hugging Face
cross-encoder id. RERANK_CANDIDATES (default 12): only the top N fused candidates are rescored; the rest keep their
fused order below them (decision 9: CPU cost grows with N).
Vectors are cached on disk (data/processed/embeddings/) and in the store, so the corpus is embedded once per model.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from app.retrieval.chunking import ROOT, Chunk, tokenize

CACHE = ROOT / "data" / "processed" / "embeddings"
ACCESS_DEFAULT = ("public", "internal")


# ------------------------------------------------------------------ filters
@dataclass
class Filters:
    """Pre-search metadata filters.
    vendor_ids: when set, keep chunks of these vendors plus vendor-neutral chunks (policies, regulations, guidance);
                drop every other vendor's documents.
    access_classes: allowed access classes; 'restricted' is never retrieved unless listed here explicitly.
    include_superseded: False drops SUPERSEDED documents (keep True when a question may hinge on version history).
    """
    vendor_ids: tuple[str, ...] | None = None
    access_classes: tuple[str, ...] = ACCESS_DEFAULT
    include_superseded: bool = True
    slices: tuple[str, ...] | None = None
    doc_types: tuple[str, ...] | None = None

    def allows(self, c: Chunk) -> bool:
        if c.access_class not in self.access_classes:
            return False
        if self.vendor_ids is not None and c.vendor_id is not None and c.vendor_id not in self.vendor_ids:
            return False
        if not self.include_superseded and c.status == "SUPERSEDED":
            return False
        if self.slices is not None and c.slice not in self.slices:
            return False
        if self.doc_types is not None and c.doc_type not in self.doc_types:
            return False
        return True

    def mask(self, chunks: list[Chunk]) -> np.ndarray:
        return np.fromiter((self.allows(c) for c in chunks), dtype=bool, count=len(chunks))


NO_FILTER = Filters(access_classes=("public", "internal", "restricted"))


def _top(scores: np.ndarray, mask: np.ndarray | None, k: int) -> np.ndarray:
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


# ------------------------------------------------------------------ sparse
class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 20, filters: Filters | None = None) -> list[tuple[Chunk, float]]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64)
        top = _top(scores, filters.mask(self.chunks) if filters else None, k)
        return [(self.chunks[i], float(scores[i])) for i in top]


# ------------------------------------------------------------------ dense
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Embedder:
    model_id: str
    encode_docs: Callable[[list[str]], np.ndarray]
    encode_query: Callable[[str], np.ndarray]
    batch: int = 64
    encode_chunks: Callable[[list[Chunk]], np.ndarray] | None = None   # backends that use title/metadata separately
    estimate: Callable[[list[Chunk]], str] | None = None               # printed before a long embedding run


def hash_embedder(dim: int = 512) -> Embedder:
    """Hashed bag of words, L2-normalised. Lexical, not semantic: for tests and offline CI only."""
    def enc(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[r, h % dim] += 1.0 if (h >> 31) & 1 else -1.0
        return out
    return Embedder(f"hash{dim}", enc, lambda q: enc([q])[0])


def gemini_embedder() -> Embedder:
    from app.retrieval.gemini_embed import GeminiEmbedder, doc_title, estimate_tokens
    g = GeminiEmbedder()

    def estimate(chunks):
        toks = sum(estimate_tokens(doc_title(c)) + estimate_tokens(c.text) + 8 for c in chunks)
        reqs = max(1, -(-toks // g.max_batch_tokens), -(-len(chunks) // g.max_batch))
        mins = max(toks / (g.limiter.tpm * g.limiter.safety), reqs / (g.limiter.rpm * g.limiter.safety))
        return (f"Gemini {g.model} ({g.dim} dims): {len(chunks):,} chunks, about {toks:,} tokens in about {reqs:,} requests; "
                f"at {g.limiter.tpm:,} tokens/min and {g.limiter.rpm} requests/min that is about {mins:.0f} min")

    return Embedder(g.model_id, lambda texts: g.embed_documents([("", t) for t in texts]), g.embed_query, batch=200,
                    encode_chunks=lambda cs: g.embed_documents([(doc_title(c), c.text) for c in cs]), estimate=estimate)


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1, n)


def endpoint_embedder() -> Embedder:
    from app.llm_client import embed
    model_name = os.environ["EMBED_MODEL"].strip()
    prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() and "-en" in model_name.lower() else ""
    enc = lambda texts: _l2(embed(list(texts), tag="retrieval"))
    model_id = os.getenv("EMBED_REUSE_VECTORS_FROM", "").strip() or f"endpoint_{model_name}"
    return Embedder(model_id, enc, lambda q: enc([prefix + q])[0], batch=int(os.getenv("EMBED_BATCH", "64")))


def get_embedder() -> Embedder:
    backend = os.getenv("EMBED_BACKEND", "gemini").strip().lower()
    if backend == "hash":
        return hash_embedder()
    if backend == "gemini":
        return gemini_embedder()
    if backend == "endpoint":
        return endpoint_embedder()
    if backend == "local":
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBED_LOCAL_MODEL", "BAAI/bge-small-en-v1.5").strip()
        m = SentenceTransformer(model_name, device="cpu")
        prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() and "-en" in model_name.lower() else ""
        enc = lambda texts: m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return Embedder(f"local_{model_name}", enc, lambda q: enc([prefix + q])[0])
    raise ValueError(f"EMBED_BACKEND={backend!r}: use gemini, local, endpoint or hash")


def cache_path(chunks: list[Chunk], model_id: str):
    key = hashlib.sha1(("\n".join(c.chunk_id + "\x1f" + c.index_text for c in chunks) + model_id).encode()).hexdigest()[:12]
    return CACHE / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', model_id)}_{key}.npy"


def embeddings_ready(chunks: list[Chunk], model_id: str, store=None) -> bool:
    """True when every chunk already has a vector (disk cache or store): dense search needs no embedding calls."""
    if cache_path(chunks, model_id).exists():
        return True
    return bool(store) and len(store.load_embeddings(model_id, [c.chunk_id for c in chunks])) == len(chunks)


def embed_chunks(chunks: list[Chunk], embedder: Embedder, store=None, progress: bool = True) -> np.ndarray:
    """Vectors for chunks (rows in the same order), from the store or the on-disk cache when present."""
    path = cache_path(chunks, embedder.model_id)
    if path.exists():
        return np.load(path)
    have = store.load_embeddings(embedder.model_id, [c.chunk_id for c in chunks]) if store else {}
    todo = [i for i, c in enumerate(chunks) if c.chunk_id not in have]
    vecs: dict[int, np.ndarray] = {i: have[c.chunk_id] for i, c in enumerate(chunks) if c.chunk_id in have}
    t0, done = time.time(), 0
    if todo and progress and embedder.estimate:
        print("  " + embedder.estimate([chunks[i] for i in todo]), flush=True)
    for s in range(0, len(todo), embedder.batch):
        idx = todo[s:s + embedder.batch]
        if embedder.encode_chunks:
            out = np.asarray(embedder.encode_chunks([chunks[i] for i in idx]), dtype=np.float32)
        else:
            out = np.asarray(embedder.encode_docs([chunks[i].index_text for i in idx]), dtype=np.float32)
        for i, v in zip(idx, out):
            vecs[i] = v
        if store:
            store.put_embeddings(embedder.model_id, {chunks[i].chunk_id: v for i, v in zip(idx, out)})
        done += len(idx)
        if progress and ((done // embedder.batch) % 20 == 0 or embedder.encode_chunks):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  embedded {done:,}/{len(todo):,} chunks ({rate:.0f}/s, ~{(len(todo) - done) / max(rate, 1e-6) / 60:.1f} min left)",
                  flush=True)
    arr = np.vstack([vecs[i] for i in range(len(chunks))]) if chunks else np.zeros((0, 1), dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, store=None, vecs: np.ndarray | None = None):
        self.chunks = chunks
        self.embedder = embedder or get_embedder()
        self.model_id = self.embedder.model_id
        v = vecs if vecs is not None else embed_chunks(chunks, self.embedder, store)
        self.vecs = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def search(self, query: str, k: int = 20, filters: Filters | None = None) -> list[tuple[Chunk, float]]:
        q = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = self.vecs @ q
        top = _top(scores, filters.mask(self.chunks) if filters else None, k)
        return [(self.chunks[i], float(scores[i])) for i in top]


# ------------------------------------------------------------------ hybrid
def rrf(rankings: Iterable[list[Chunk]], k: int = 60) -> list[tuple[Chunk, float]]:
    """Reciprocal rank fusion: score = sum over rankings of 1 / (k + rank)."""
    score, keep = {}, {}
    for ranking in rankings:
        for r, c in enumerate(ranking, start=1):
            score[c.chunk_id] = score.get(c.chunk_id, 0.0) + 1.0 / (k + r)
            keep[c.chunk_id] = c
    return sorted(((keep[i], s) for i, s in score.items()), key=lambda x: -x[1])


DENSE_FALLBACKS: list[str] = []      # queries answered by BM25 alone because dense search failed (reported, never hidden)


class HybridRetriever:
    """Top-n from BM25 and top-n from dense, merged with reciprocal rank fusion."""
    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, n_each: int = 20, rrf_k: int = 60):
        self.bm25, self.dense, self.n_each, self.rrf_k = bm25, dense, n_each, rrf_k
        self.chunks = bm25.chunks

    def search(self, query: str, k: int = 40, filters: Filters | None = None) -> list[tuple[Chunk, float]]:
        a = [c for c, _ in self.bm25.search(query, self.n_each, filters)]
        try:
            b = [c for c, _ in self.dense.search(query, self.n_each, filters)]
        except Exception as e:                                # embeddings endpoint slow or down: keyword results, not a failure
            DENSE_FALLBACKS.append(f"{type(e).__name__}: {str(e)[:120]}")
            print(f"[retrieval] dense search failed ({type(e).__name__}); BM25 only for this query", flush=True)
            return [(c, 1.0 / (self.rrf_k + i + 1)) for i, c in enumerate(a)][:k]
        return rrf([a, b], self.rrf_k)[:k]


# ------------------------------------------------------------------ passages, dedupe, rerank
@dataclass
class Passage:
    """What the copilot reads and cites. For small2big the text is the parent section, not the small hit."""
    passage_id: str
    doc_id: str
    title: str
    text: str
    vendor_id: str | None
    doc_type: str
    document_date: str | None
    status: str
    access_class: str
    source_id: str
    heading: str = ""
    score: float = 0.0
    retrieval_rank: int = 0
    rerank_score: float | None = None
    matched_chunk_ids: list[str] = field(default_factory=list)

    def citation(self) -> dict:
        return {"doc_id": self.doc_id, "passage_id": self.passage_id, "title": self.title, "vendor_id": self.vendor_id,
                "document_date": self.document_date, "evidence_type": self.doc_type, "status": self.status}


def to_passages(hits: list[tuple[Chunk, float]], parents: dict[str, Chunk] | None = None,
                max_per_doc: int | None = None) -> list[Passage]:
    """Map hits to passages (child -> parent for small2big), dedupe by passage (keep the best rank) and optionally
    cap passages per document, so one long document cannot fill every slot."""
    out, seen, per_doc = [], {}, {}
    for c, s in hits:
        p = parents.get(c.parent_id) if (parents and c.parent_id) else None
        src = p or c
        if src.chunk_id in seen:
            seen[src.chunk_id].matched_chunk_ids.append(c.chunk_id)
            continue
        if max_per_doc and per_doc.get(src.doc_id, 0) >= max_per_doc:
            continue
        per_doc[src.doc_id] = per_doc.get(src.doc_id, 0) + 1
        ps = Passage(src.chunk_id, src.doc_id, src.title, src.text, src.vendor_id, src.doc_type, src.document_date, src.status,
                     src.access_class, src.source_id, src.heading, s, len(out) + 1, None, [c.chunk_id])
        seen[src.chunk_id] = ps
        out.append(ps)
    return out


RERANK_MODELS = {"minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2", "bge-base": "BAAI/bge-reranker-base"}


def rerank_model_name(name: str | None = None) -> str:
    n = (name or os.getenv("RERANK_MODEL", "minilm")).strip()
    return RERANK_MODELS.get(n.lower(), n)


def rerank_candidates() -> int:
    return max(1, int(os.getenv("RERANK_CANDIDATES", "12")))


class Reranker:
    """Cross-encoder reranker on CPU (sentence-transformers CrossEncoder)."""

    def __init__(self, model_name: str | None = None, max_chars: int = 2000):
        from sentence_transformers import CrossEncoder
        self.model_name = rerank_model_name(model_name)
        self.model = CrossEncoder(self.model_name, device="cpu", max_length=512)
        self.max_chars = max_chars

    def score(self, query: str, passages: list[Passage]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, f"{p.title}. {p.heading}\n{p.text}"[: self.max_chars]) for p in passages]
        return [float(x) for x in self.model.predict(pairs, batch_size=16, show_progress_bar=False)]


def get_reranker():
    """RERANK_BACKEND: cross_encoder (default; needs sentence-transformers), auto (cross-encoder if sentence-transformers is
    installed, else none: used by the API container, which ships without PyTorch), or none."""
    backend = os.getenv("RERANK_BACKEND", "cross_encoder").strip().lower()
    if backend in ("none", "off", "0"):
        return None
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return None
        return Reranker()
    if backend == "cross_encoder":
        return Reranker()
    raise ValueError(f"RERANK_BACKEND={backend!r}: use cross_encoder, auto or none")


def rerank(query: str, passages: list[Passage], reranker, keep: int | None = None, candidates: int | None = None) -> list[Passage]:
    """Rescore the top `candidates` passages (all if None) with the cross-encoder; the rest keep their order below them."""
    if reranker is None:
        return passages[:keep] if keep else passages
    head, tail = (passages[:candidates], passages[candidates:]) if candidates else (passages, [])
    for p, s in zip(head, reranker.score(query, head)):
        p.rerank_score = s
    ranked = sorted(head, key=lambda p: -p.rerank_score) + tail
    return ranked[:keep] if keep else ranked


class RetrievalPipeline:
    """The production path the tools use: filters -> hybrid (BM25 top 20 + dense top 20, RRF) -> passages (dedupe,
    child -> parent) -> cross-encoder rerank -> top `keep` passages with citation metadata."""

    def __init__(self, chunks: list[Chunk], parents: dict[str, Chunk] | None = None, embedder: Embedder | None = None,
                 reranker="auto", store=None, n_each: int = 20, keep: int = 6, max_per_doc: int | None = 3,
                 use_dense: bool = True):
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks, embedder, store) if use_dense else None
        self.hybrid = HybridRetriever(self.bm25, self.dense, n_each=n_each) if use_dense else None
        self.n_each = n_each
        self.mode = "hybrid" if use_dense else "bm25"
        self.parents = parents
        self.rerank_note = ""
        if reranker == "auto":
            try:
                self.reranker = get_reranker()
            except Exception as e:                    # model not downloaded (HF_HUB_OFFLINE=1), no torch, ...: never fail search
                if os.getenv("RERANK_REQUIRED", "0") == "1":
                    raise
                self.reranker = None
                self.rerank_note = f"reranker off: {type(e).__name__}: {str(e)[:160]}"
                print(f"[retrieval] {self.rerank_note}")
        else:
            self.reranker = reranker
        self.rerank_candidates = rerank_candidates()
        self.keep, self.max_per_doc = keep, max_per_doc

    def search(self, query: str, filters: Filters | None = None, keep: int | None = None) -> list[Passage]:
        filters = filters or Filters()
        hits = (self.hybrid.search(query, 2 * self.n_each, filters) if self.hybrid
                else self.bm25.search(query, 2 * self.n_each, filters))
        candidates = to_passages(hits, self.parents, self.max_per_doc)
        return rerank(query, candidates, self.reranker, keep or self.keep, self.rerank_candidates)
