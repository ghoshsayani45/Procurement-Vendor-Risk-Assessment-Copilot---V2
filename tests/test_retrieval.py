"""chunking, store (idempotent ingestion), filters, hybrid fusion, passages, metrics.
All offline: the hash embedder and a fake reranker stand in for the real models."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.retrieval.chunking import (CORPUS, STRATEGIES, chunk_corpus, fixed_chunks, is_heading, load_corpus, parents_for,
                                    sections, small2big_chunks, structure_chunks, tokenize)
from app.retrieval.retrievers import (BM25Retriever, DenseRetriever, Filters, HybridRetriever, Passage, RetrievalPipeline,
                                      hash_embedder, rerank, rrf, to_passages)

needs_corpus = pytest.mark.skipif(not CORPUS.exists(), reason="corpus_v1.jsonl not built (make corpus)")


def doc(doc_id="D1", text="", vendor_id=None, **kw):
    base = dict(doc_id=doc_id, text=text, vendor_id=vendor_id, slice="contracts", title=f"Title {doc_id}", doc_type="contract",
                access_class="internal", status="CURRENT", document_date="2026-01-01", source_id="SYN", sha1="x", words=0)
    base.update(kw)
    return base


CONTRACT = """# SaaS Agreement

**Between** Customer and Supplier.

## 1. Services
The Supplier will provide CRM services as described in Schedule 1 to the customer business owner.

## 2. Term and renewal
The term renews automatically for one year unless a party gives notice not less than 120 days before expiry.

## 5. Limitation of liability
The Supplier's maximum liability is limited to an amount equal to 24 months' fees under this Agreement.
"""


# ------------------------------------------------------------------ chunking
def test_heading_detection():
    for h in ["## 5. Limitation of liability", "4.1 Required evidence", "Rule 144 Fundamental principles", "DEFINITIONS",
              "TERM AND TERMINATION", "Chapter 3: Procurement of goods", "IAM-02 Strong Password Policy", "A&A-01 Audit policy"]:
        assert is_heading(h), h
    for not_h in ["The Supplier shall notify the Customer of any breach within 72 hours.", "| SOC 2 | Required |", "II", "12",
                  "**Status:** Executed agreement", ""]:
        assert not is_heading(not_h), not_h


def test_structure_chunks_keep_clauses_whole_and_cover_text():
    d = doc(text=CONTRACT)
    secs = sections(CONTRACT)
    assert [h for h, _ in secs][-1] == "5. Limitation of liability"
    chunks = structure_chunks(d, max_tokens=60, min_tokens=10)
    assert len(chunks) >= 2
    joined = " ".join(c.text for c in chunks).split()
    assert joined == CONTRACT.split()                                    # nothing lost, nothing duplicated
    liab = next(c for c in chunks if "24 months" in c.text)
    assert "Limitation of liability" in liab.text and liab.index_text.startswith("[Title D1")


def test_oversized_section_split_and_tail_folded():
    body = "\n\n".join(" ".join(f"w{i}_{j}" for j in range(120)) for i in range(5))   # 600 words, no headings
    chunks = structure_chunks(doc(text=body), max_tokens=400)
    assert all(c.tokens <= 500 for c in chunks)                         # <= max_tokens * 1.25
    assert min(len(c.text.split()) for c in chunks) > 40                 # no tiny tail chunk


def test_small2big_children_inside_parents():
    d = doc(text=CONTRACT * 6)
    children, parents = small2big_chunks(d, child_tokens=40, overlap_tokens=8)
    pmap = {p.chunk_id: p for p in parents}
    assert all(c.parent_id in pmap and c.text in " ".join(pmap[c.parent_id].text.split()) for c in children)


def test_fixed_windows_overlap_and_ids_distinct():
    d = doc(text=" ".join(f"t{i}" for i in range(1000)))
    a, b = fixed_chunks(d), fixed_chunks(d, header=True)
    assert len(a) == 3 and a[0].text.split()[-48:] == a[1].text.split()[:48]          # 64-token overlap = 48 words
    assert not {c.chunk_id for c in a} & {c.chunk_id for c in b}
    assert b[0].index_text.startswith("[Title D1]") and b[0].text == a[0].text


@needs_corpus
def test_all_strategies_on_corpus():
    docs = load_corpus()[:60]
    for s in STRATEGIES:
        chunks = chunk_corpus(docs, s)
        assert chunks and len({c.chunk_id for c in chunks}) == len(chunks), s
        assert {c.doc_id for c in chunks} == {d["doc_id"] for d in docs}, s
    assert max(c.tokens for c in chunk_corpus(docs, "structure")) <= 600


# ------------------------------------------------------------------ filters, fusion, passages
def _mini_index():
    docs = [doc("A1", "liability cap twelve months fees supplier", vendor_id="V001"),
            doc("B1", "liability cap twenty four months fees supplier", vendor_id="V002"),
            doc("P1", "policy liability cap minimum twelve months", slice="procurement_policy", doc_type="internal_policy"),
            doc("R1", "restricted board minutes liability cap", access_class="restricted"),
            doc("S1", "old policy liability cap six months", status="SUPERSEDED", doc_type="internal_policy")]
    return docs, chunk_corpus(docs, "fixed512_ctx")


def test_filters_applied_before_search():
    docs, chunks = _mini_index()
    bm = BM25Retriever(chunks)
    got = {c.doc_id for c, _ in bm.search("liability cap", 10, Filters(vendor_ids=("V001",)))}
    assert got == {"A1", "P1", "S1"}                                      # other vendor and restricted never returned
    got = {c.doc_id for c, _ in bm.search("liability cap", 10, Filters(include_superseded=False))}
    assert "S1" not in got and "R1" not in got and {"A1", "B1", "P1"} <= got
    assert "R1" in {c.doc_id for c, _ in bm.search("liability cap", 10, Filters(access_classes=("restricted",)))}


def test_rrf_and_hybrid():
    docs, chunks = _mini_index()
    a, b, c = chunks[0], chunks[1], chunks[2]
    fused = rrf([[a, b], [b, c]], k=60)
    assert fused[0][0] is b and {x.chunk_id for x, _ in fused} == {a.chunk_id, b.chunk_id, c.chunk_id}
    h = HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks, hash_embedder()), n_each=3)
    res = h.search("twelve months liability", 10, Filters())
    assert res and len({x.chunk_id for x, _ in res}) == len(res) and "R1" not in {x.doc_id for x, _ in res}


def test_passages_dedupe_parent_mapping_and_cap():
    d = doc(text=CONTRACT * 4)
    children, parents = small2big_chunks(d, child_tokens=30, overlap_tokens=5)
    pmap = {p.chunk_id: p for p in parents}
    ps = to_passages([(c, 1.0) for c in children], pmap)
    assert [p.passage_id for p in ps] == [p.chunk_id for p in parents]          # each parent once, in first-hit order
    assert sum(len(p.matched_chunk_ids) for p in ps) == len(children)
    assert len(to_passages([(c, 1.0) for c in children], pmap, max_per_doc=1)) == 1
    cit = ps[0].citation()
    assert set(cit) == {"doc_id", "passage_id", "title", "vendor_id", "document_date", "evidence_type", "status"}


class FakeReranker:
    model_name = "fake"

    def score(self, query, passages):
        q = set(tokenize(query))
        return [len(q & set(tokenize(p.text))) - 0.001 * i for i, p in enumerate(passages)]


def test_pipeline_end_to_end_offline():
    docs, chunks = _mini_index()
    pipe = RetrievalPipeline(chunks, embedder=hash_embedder(), reranker=FakeReranker(), keep=2)
    out = pipe.search("twenty four months liability cap", Filters(vendor_ids=("V002",)))
    assert 1 <= len(out) <= 2 and out[0].doc_id == "B1" and out[0].rerank_score is not None
    assert all(p.vendor_id in (None, "V002") and p.access_class != "restricted" for p in out)


def test_rerank_none_keeps_order():
    ps = [Passage(str(i), str(i), "t", "x", None, "", None, "CURRENT", "public", "") for i in range(5)]
    assert [p.passage_id for p in rerank("q", ps, None, keep=3)] == ["0", "1", "2"]


# ------------------------------------------------------------------ store and ingestion (step 2.1)
def _docs_for_store():
    import hashlib
    ds = []
    for i, (vid, text) in enumerate([("V001", CONTRACT), (None, "## 1. Policy\nAbsent evidence is treated as missing.")]):
        d = doc(f"DOC-{i}", text, vendor_id=vid, document_date=["01-Aug-2023", "2022-06"][i])
        d["sha1"] = hashlib.sha1(text.encode()).hexdigest()
        ds.append(d)
    return ds


def test_ingestion_idempotent_and_versioned(tmp_path, monkeypatch):
    from app.db.store import SQLiteStore
    from scripts import index_corpus as ic
    monkeypatch.setenv("EMBED_BACKEND", "hash")
    monkeypatch.setattr("app.retrieval.retrievers.CACHE", tmp_path / "emb")
    store = SQLiteStore(tmp_path / "s.sqlite")
    docs = _docs_for_store()
    r1 = ic.run(store, ["structure", "fixed512"], embed=True, verify=False, docs=docs)
    assert r1["documents"]["new_or_changed"] == 2 and r1["counts"]["vendors"] == 35
    n_chunks = r1["counts"]["chunks"]
    assert r1["strategies"]["structure"]["embeddings_added"] == n_chunks["structure"]
    r2 = ic.run(store, ["structure", "fixed512"], embed=True, verify=False, docs=docs)      # same corpus: nothing to do
    assert r2["documents"]["new_or_changed"] == 0 and r2["counts"]["chunks"] == n_chunks
    assert all(v["documents_rechunked"] == 0 and v["embeddings_added"] == 0 for v in r2["strategies"].values())
    changed = [dict(docs[0], text=docs[0]["text"] + "\n## 9. Audit\nOnce a year.", sha1="changed"), docs[1]]
    r3 = ic.run(store, ["structure"], verify=False, docs=changed)
    assert r3["documents"]["new_or_changed"] == 1 and r3["strategies"]["structure"]["documents_rechunked"] == 1
    r4 = ic.run(store, ["structure"], verify=False, docs=changed[:1])                        # DOC-1 dropped from corpus
    assert r4["documents"]["removed"] == 1 and r4["counts"]["documents"] == 1
    loaded = store.load_chunks("structure")
    assert {c.doc_id for c in loaded} == {"DOC-0"} and loaded[0].vendor_id == "V001" and loaded[0].document_date == "2023-08-01"
    monkeypatch.setattr(ic, "CHUNKER_VERSION", "test-bump")                                  # new chunker -> re-chunk all
    assert ic.run(store, ["structure"], verify=False, docs=changed[:1])["strategies"]["structure"]["documents_rechunked"] == 1
    assert store.counts()["runs"] == 7


def test_norm_date_and_normalise():
    from scripts.index_corpus import IngestError, norm_date, normalise
    assert norm_date("2026-01-31") == "2026-01-31" and norm_date("2022-06") == "2022-06-01"
    assert norm_date("01-Aug-2023") == "2023-08-01" and norm_date("v1 (2021)") == "2021-01-01"
    assert norm_date("r1-upd1") is None and norm_date(None) is None
    assert normalise(doc(vendor_id="v009"))["vendor_id"] == "V009"
    with pytest.raises(IngestError):
        normalise(doc(vendor_id="ACME"))
    with pytest.raises(IngestError):
        normalise(doc(access_class="secret"))


@needs_corpus
def test_corpus_matches_manifest_and_tamper_detected():
    from scripts.index_corpus import IngestError, check_against_manifest
    docs = load_corpus()
    check_against_manifest(docs)
    docs[5] = dict(docs[5], text=docs[5]["text"] + " tampered")
    with pytest.raises(IngestError, match="1 changed"):
        check_against_manifest(docs)


def test_embeddings_roundtrip(tmp_path):
    from app.db.store import SQLiteStore
    s = SQLiteStore(tmp_path / "e.sqlite")
    s.put_embeddings("m", {"c1": np.arange(4, dtype=np.float32)})
    got = s.load_embeddings("m", ["c1", "c2"])
    assert list(got) == ["c1"] and got["c1"].tolist() == [0, 1, 2, 3]


# ------------------------------------------------------------------ metrics
def test_quote_metric_is_passage_level():
    from app.retrieval.evaluation import score_query
    q = {"id": "Q", "category": "lookup_vendor", "split": "dev", "gold_doc_ids": ["D"],
         "gold_quotes": [{"doc_id": "D", "quote": "Within  72 hours of becoming aware"}]}
    mk = lambda pid, text: Passage(pid, "D", "t", text, None, "", None, "CURRENT", "internal", "")
    wrong_chunk = score_query(q, [mk("D#1", "fees and payment terms")])
    right_chunk = score_query(q, [mk("D#2", "notify the Customer **within 72 hours** of becoming aware of it")])
    assert wrong_chunk["hit@5"] == 1 and wrong_chunk["quote@5"] == 0       # right document, wrong passage
    assert right_chunk["quote@5"] == 1 and right_chunk["mrr"] == 1


def test_rerank_rescores_only_the_top_candidates():
    ps = [Passage(str(i), str(i), "t", "x", None, "", None, "CURRENT", "public", "") for i in range(6)]

    class Reverse:
        def score(self, query, passages):
            return [float(i) for i in range(len(passages))]       # later = better: reverses the head
    out = rerank("q", ps, Reverse(), candidates=3)
    assert [p.passage_id for p in out] == ["2", "1", "0", "3", "4", "5"]      # the tail keeps its fused order


def test_reranker_that_cannot_load_turns_off_instead_of_failing(monkeypatch):
    from app.retrieval import retrievers
    monkeypatch.setenv("RERANK_BACKEND", "cross_encoder")
    monkeypatch.delenv("RERANK_REQUIRED", raising=False)
    monkeypatch.setattr(retrievers, "get_reranker", lambda: (_ for _ in ()).throw(OSError("model not in the offline cache")))
    _, chunks = _mini_index()
    pipe = RetrievalPipeline(chunks, use_dense=False)
    assert pipe.reranker is None and "offline cache" in pipe.rerank_note
    assert pipe.search("liability cap", keep=2)
    monkeypatch.setenv("RERANK_REQUIRED", "1")
    with pytest.raises(OSError):
        RetrievalPipeline(chunks, use_dense=False)


def test_rerank_model_aliases(monkeypatch):
    from app.retrieval.retrievers import rerank_candidates, rerank_model_name
    monkeypatch.delenv("RERANK_MODEL", raising=False)
    monkeypatch.delenv("RERANK_CANDIDATES", raising=False)
    assert rerank_model_name() == "cross-encoder/ms-marco-MiniLM-L-6-v2" and rerank_candidates() == 12
    assert rerank_model_name("bge-base") == "BAAI/bge-reranker-base"
    assert rerank_model_name("org/custom-ce") == "org/custom-ce"


def test_rerank_experiment_verdict_follows_the_fixed_rule():
    from scripts.rerank_experiment import verdict
    base = {"hit@5": 0.9, "mrr": 0.7, "quote@5": 0.60, "latency_p95_ms": 100}
    rep = {"variants": {"none": base,
                        "minilm@12": {"hit@5": 0.9, "mrr": 0.72, "quote@5": 0.64, "latency_p95_ms": 900},
                        "bge-base@12": {"hit@5": 0.92, "mrr": 0.75, "quote@5": 0.70, "latency_p95_ms": 2500},
                        "minilm@20": {"hit@5": 0.88, "mrr": 0.73, "quote@5": 0.66, "latency_p95_ms": 1200}}}
    v = verdict(rep)
    assert v["adopt"] == "minilm@12" and v["env"]["RERANK_CANDIDATES"] == "12"
    assert rep["variants"]["bge-base@12"]["checks"]["latency"] is False and rep["variants"]["minilm@20"]["checks"]["hit@5"] is False
    rep["variants"]["minilm@12"]["quote@5"] = 0.61
    assert verdict(rep)["adopt"] is None and verdict(rep)["env"] == {"RERANK_BACKEND": "none"}
    assert "nothing was measured" in verdict({"variants": {"none": base, "minilm@12": {"error": "no torch"}}})["reason"]


def test_endpoint_embedder_prefixes_bge_queries_normalises_and_can_reuse_vectors(monkeypatch):
    from app import llm_client
    from app.retrieval import retrievers
    seen = []

    def fake_embed(texts, tag=""):
        seen.extend(texts)
        return [[3.0, 4.0] for _ in texts]
    monkeypatch.setattr(llm_client, "embed", fake_embed)
    monkeypatch.setenv("EMBED_BACKEND", "endpoint")
    monkeypatch.setenv("EMBED_MODEL", "bge-small-en-v1.5")
    monkeypatch.delenv("EMBED_REUSE_VECTORS_FROM", raising=False)
    e = retrievers.get_embedder()
    assert e.model_id == "endpoint_bge-small-en-v1.5"
    q = e.encode_query("payment terms")
    assert seen[-1] == retrievers.BGE_QUERY_PREFIX + "payment terms" and abs(float((q ** 2).sum()) - 1) < 1e-6
    e.encode_docs(["a passage"])
    assert seen[-1] == "a passage"                                        # documents get no prefix
    monkeypatch.setenv("EMBED_REUSE_VECTORS_FROM", "local_BAAI/bge-small-en-v1.5")
    assert retrievers.get_embedder().model_id == "local_BAAI/bge-small-en-v1.5"


def test_embed_ca_bundle_must_exist_and_quotas_are_separate(monkeypatch, tmp_path):
    from app import llm_client
    monkeypatch.setenv("EMBED_CA_BUNDLE", str(tmp_path / "missing.crt"))
    with pytest.raises(FileNotFoundError):
        llm_client._verify("EMBED")
    ca = tmp_path / "vm02-ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setenv("EMBED_CA_BUNDLE", str(ca))
    monkeypatch.setenv("LLM_CA_BUNDLE", "")
    assert llm_client._verify("EMBED") == str(ca)
    from app.retrieval.chunking import ROOT
    monkeypatch.setenv("EMBED_CA_BUNDLE", "certs/does-not-exist.crt")
    with pytest.raises(FileNotFoundError, match=str(ROOT / "certs").replace("\\", "\\\\")):
        llm_client._verify("EMBED")                                       # relative = from the project folder
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"kind": "chat", "total_tokens": 100}\n{"kind": "embed", "total_tokens": 5000}\n')
    monkeypatch.setattr(llm_client, "LEDGER", ledger)
    assert llm_client.tokens_used("chat") == 100 and llm_client.tokens_used("embed") == 5000
    monkeypatch.setenv("TEAM_TOKEN_QUOTA", "1000")
    llm_client._guard()                                                   # 5k embedding tokens do not count against chat
    monkeypatch.setenv("EMBED_TOKEN_QUOTA", "4000")
    with pytest.raises(llm_client.QuotaExceeded):
        llm_client._guard(kind="embed")


def test_embed_splits_requests_to_the_endpoint_batch_limit(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app import llm_client
    sizes = []

    class FakeEmbeddings:
        def create(self, model, input):
            assert len(input) <= 16, "endpoint rejects batches over 16"
            sizes.append(len(input))
            data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in reversed(list(enumerate(input)))]
            return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(input), completion_tokens=0,
                                                                     total_tokens=len(input)))
    monkeypatch.setattr(llm_client, "_client", lambda env: SimpleNamespace(embeddings=FakeEmbeddings()))
    monkeypatch.setattr(llm_client, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setenv("EMBED_MODEL", "bge-small-en-v1.5")
    monkeypatch.delenv("EMBED_MAX_BATCH", raising=False)
    texts = ["x" * (i + 1) for i in range(40)]
    out = llm_client.embed(texts)
    assert sizes == [16, 16, 8] and [v[0] for v in out] == [float(i + 1) for i in range(40)]   # order kept


def _fake_endpoint(monkeypatch, tmp_path, limit=512, tokens=lambda t: len(t.split())):
    """A TEI-like server: at most 16 inputs, rejects any input over `limit` tokens with TEI's 422 message."""
    from types import SimpleNamespace
    from app import llm_client
    calls = []

    class FakeEmbeddings:
        def create(self, model, input):
            calls.append(list(input))
            assert len(input) <= 16
            for t in input:
                if tokens(t) + 2 > limit:
                    raise RuntimeError(f"Error code: 422 - Input validation error: `inputs` must have less than {limit} "
                                       f"tokens. Given: {tokens(t) + 2}")
            data = [SimpleNamespace(index=i, embedding=[float(tokens(t))]) for i, t in enumerate(input)]
            return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1))
    monkeypatch.setattr(llm_client, "_client", lambda env: SimpleNamespace(embeddings=FakeEmbeddings()))
    monkeypatch.setattr(llm_client, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setenv("EMBED_MODEL", "bge-small-en-v1.5")
    return calls


def test_long_inputs_are_clipped_to_510_word_pieces_like_sentence_transformers(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app import llm_client

    class WordTokenizer:                                   # stand-in for bge's tokenizer: one piece per word
        def encode_batch(self, texts, add_special_tokens=False):
            out = []
            for t in texts:
                spans, pos = [], 0
                for w in t.split(" "):
                    spans.append((pos, pos + len(w)))
                    pos += len(w) + 1
                out.append(SimpleNamespace(ids=list(range(len(spans))), offsets=spans))
            return out
    monkeypatch.setattr(llm_client, "_embed_tokenizer", lambda: WordTokenizer())
    calls = _fake_endpoint(monkeypatch, tmp_path)
    long = " ".join(f"w{i}" for i in range(900))
    out = llm_client.embed([long, "short text"])
    assert out == [[510.0], [2.0]] and len(calls) == 1                     # one request, nothing rejected
    assert calls[0][0].endswith("w509")                                     # exactly the first 510 pieces kept


def test_without_a_tokenizer_the_server_count_guides_the_cut(monkeypatch, tmp_path):
    from app import llm_client
    monkeypatch.setattr(llm_client, "_embed_tokenizer", lambda: None)
    calls = _fake_endpoint(monkeypatch, tmp_path)
    texts = ["short"] * 3 + [" ".join(["word"] * 904)]
    out = llm_client.embed(texts)
    assert len(out) == 4 and out[:3] == [[1.0]] * 3 and 400 < out[3][0] <= 510
    assert len(calls) <= 8                                                  # batch, then one by one, then the cut


def test_hybrid_search_falls_back_to_bm25_when_the_embeddings_endpoint_fails():
    from app.retrieval import retrievers
    docs, chunks = _mini_index()
    emb = hash_embedder()
    dense = DenseRetriever(chunks, emb)

    def down(q):
        raise TimeoutError("embeddings endpoint read timeout")
    dense.embedder = retrievers.Embedder(emb.model_id, emb.encode_docs, down)
    h = HybridRetriever(BM25Retriever(chunks), dense, n_each=5)
    before = len(retrievers.DENSE_FALLBACKS)
    hits = h.search("liability cap", 5)
    assert hits and len(retrievers.DENSE_FALLBACKS) == before + 1


def test_embedding_requests_have_a_short_timeout_and_one_retry(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app import llm_client
    attempts = []

    class ReadTimeout(Exception):
        pass

    class FakeEmbeddings:
        def create(self, model, input):
            attempts.append(1)
            if len(attempts) == 1:
                raise ReadTimeout("read timed out")
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])],
                                   usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1))
    monkeypatch.setattr(llm_client, "_client", lambda env: SimpleNamespace(embeddings=FakeEmbeddings()))
    monkeypatch.setattr(llm_client, "_embed_tokenizer", lambda: None)
    monkeypatch.setattr(llm_client, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setenv("EMBED_MODEL", "bge-small-en-v1.5")
    assert llm_client.embed(["q"]) == [[1.0]] and len(attempts) == 2
