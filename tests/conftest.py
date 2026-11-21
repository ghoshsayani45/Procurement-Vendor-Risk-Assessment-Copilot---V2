"""Test-wide isolation: traces go to a temp file, never to your Langfuse; tests never use your real API keys."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_observability(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_LOG_PATH", str(tmp_path / "traces.jsonl"))
    monkeypatch.setenv("TRACING", "on")
    # SET, not delete: modules call load_dotenv() when first imported (often inside a test, after this fixture), and
    # load_dotenv never overrides a variable that exists, so a value set here wins over the developer's .env
    for var, safe in {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "LLM_CONSTRAINED": "", "FAULT_INJECT": "",
                      "API_DEMO_FAULTS": "0", "API_KEY": "", "LLM_CACHE_SALT": "", "AGENT_KEY_FACTS": "1",
                      "AGENT_CITATION_ENUM": "1", "AGENT_CRITIC": "1", "AGENT_JSON_RETRY": "1"}.items():
        monkeypatch.setenv(var, safe)
    monkeypatch.setenv("LLM_CACHE", "0")              # .env may turn the response cache on for eval runs; tests never use it
    import app.observability.tracing as tr
    monkeypatch.setattr(tr, "_client", None)
    monkeypatch.setattr(tr, "_client_failed", False)
    yield


@pytest.fixture(scope="session")
def corpus_store(tmp_path_factory):
    """A real store built once from corpus v1 (structure chunks, vendor records, no embeddings): retrieval runs as BM25."""
    from app.retrieval.chunking import CORPUS
    if not CORPUS.exists():
        pytest.skip("corpus_v1.jsonl not built (make corpus)")
    from app.db.store import SQLiteStore
    from scripts import index_corpus
    s = SQLiteStore(tmp_path_factory.mktemp("store") / "corpus.sqlite")
    index_corpus.run(s, ["structure"], verify=False)
    return s


@pytest.fixture
def tool_ctx(corpus_store, monkeypatch):
    """ToolContext on corpus_store with an in-memory audit log (ctx.extras['events'])."""
    from app.tools.base import ToolContext
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "structure")
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("RERANK_BACKEND", "none")
    events = []                                   # the pipeline is cached per store + settings, so it is built once
    yield ToolContext(store=corpus_store, audit=lambda action, detail: events.append((action, detail)), extras={"events": events})
