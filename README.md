# Procurement & Vendor Risk Assessment Copilot

A Track B capstone (Executive PG Programme in Generative AI and Agentic AI). It is a multi-agent copilot that assesses technology vendors for onboarding and renewal, built on a hosted ≤8B model. It retrieves cited evidence, applies a deterministic risk rubric, and routes every decision to a human approver.

## Status
| Week 1 step | Status | Where |
|---|---|---|
| Day 0 setup | Done (endpoint values still to fill in `.env`) | `.env.example`, `app/llm_client.py`, `docker/`, `app/db/schema.sql` |
| 1.1 Task specification | Done; decisions D1–D15 need team sign-off | `docs/spec.md`, `docs/decisions.md` |
| 1.2 Rubric and schemas | Done | `app/scoring/`, `app/schemas.py`, `schemas/` |
| 1.3 Synthetic vendor set | Done (35 vendors) | `data/ground_truth/` |
| 1.4 Corpus | Done; **corpus v1 frozen: 1,053 documents** | `data/synthetic/`, `data/manifests/`, `scripts/ingest.py` |
| 1.5 Evaluation set | Done; **eval_v1 frozen: 80 queries (30 dev / 50 test)**, with grader | `data/evaluation/`, `scripts/grade.py` |
| 1.6–1.7 Prompting baseline + failure log | Harness built and tested offline; **run needs the course endpoint** (`make prompt-baseline`) | `app/prompting.py`, `app/failure_analysis.py`, `docs/failure_log.md` |
| 1.8 Retrieval baselines | **BM25 done** (hit@10 dev 0.89 / test 0.95; ACORD NDCG@5 0.07); dense waits for the embeddings endpoint | `docs/baselines.md`, `reports/retrieval_baseline_v1.json` |
| **Week 2** | | |
| 2.1 Ingestion into the store | **Done**: idempotent, versioned; SQLite by default, Postgres + pgvector optional (D18) | `app/db/store.py`, `scripts/index_corpus.py` (`make index`) |
| 2.2 Chunking experiments | **BM25 run done** (4 strategies); dense/hybrid run needs embeddings on a laptop | `app/retrieval/chunking.py`, `docs/experiments/chunking.md` |
| 2.3 Hybrid retrieval + reranker | **Built**: BM25 + dense (Gemini embeddings, D23), RRF, dedupe, cross-encoder, pre-search filters, cited passages (D21) | `app/retrieval/retrievers.py`, `app/retrieval/service.py` |
| 2.4 Retrieval table | Script done; **full table needs the Gemini embedding run** (`make retrieval-eval`) | `scripts/retrieval_eval.py`, `docs/experiments/retrieval.md` |
| 2.5 Five tools | **Done**: read-only, strict schemas, timeouts, audit + trace per call; all 35 vendors match the ground truth (D24, D25) | `app/tools/`, `tests/test_tools.py` |
| 2.6 Schema + API | **Done**: Alembic migrations, FastAPI routes, API keys, rate limit, human-only decisions; Docker service added (D26). **Run `make up` to check the gate** | `app/api/`, `migrations/`, `docker/` |
| 2.7 Tracing + threat model | **Done**: Langfuse v2 + local traces (D27), `docs/threat_model.md`, safety suite (49 tests), Streamlit wireframes. **Langfuse keys needed for the gate trace** | `app/observability/`, `app/safety.py`, `tests/test_safety.py`, `app/ui/` |
| **Week 3** | | |
| 3.1–3.2 Agent graph + planner | **Built**: LangGraph validate → plan → execute → write → critic → finalize, one replan; template plans, LLM fills search queries only (D30, D31) | `app/agents/` |
| 3.3 Reliability layer | **Built**: constrained decoding if the endpoint supports it (`make check-constrained`), JSON repair, one retry, safe defaults, step cap 10, token budgets, context shrink, response cache (D32, D33) | `app/agents/llm.py`, `app/llm_client.py` |
| 3.4 Critic | **Built**: code checks for citations, grounded numbers, band words, approvals, PII, schema (D34) | `app/agents/nodes.py` |
| 3.5 Persistence + API | **Built**: per-review checkpoints (SQLite / Postgres); API reviews and messages run the agent, tools as fallback (D35) | `app/agents/graph.py`, `app/api/main.py` |
| 3.6 Routing and refusals | **Built**: rules first, LLM router only for risky unmatched questions; paraphrase probes in tests (D36) | `app/agents/rules.py`, `tests/test_agent.py` |
| 3.7 UI | **Built**: one-click demo paths, plan view, checklist by status, scorecard with weights, reviewer override (human key, reason required), approval panel, Markdown export | `app/ui/`, `app/api/main.py` |
| 3.8–3.9 Agent eval + failure taxonomy | Script done; **real run needs the endpoint** (`make agent-smoke`, then `make agent-eval`) | `scripts/agent_eval.py`, `app/agents/taxonomy.py`, `docs/agent_eval.md` |
| **Week 4** | | |
| Failure-pattern fixes (D40) | **Built**: key facts first, citations limited to real ids, the named regulation first, no drafts in rule answers; measured on dev only | `app/agents/key_facts.py`, `glossary.py`, `nodes.py` |
| 4.1 Benchmark | Script done: 3 dev runs with fresh calls for variance, targets table, usefulness judge, 20-answer human calibration (`make benchmark`) | `scripts/benchmark.py`, `docs/benchmark.md` |
| 4.3 Safety regression | Script done: corpus checks, 20 new probes, test suites per threat (`make safety-report`); its first pass found and fixed 4 gaps (D41) | `scripts/safety_report.py`, `docs/safety_report.md` |
| 4.4 Ablations | Script done: switches for key facts, citation limit, critic, JSON retry, constrained decoding, reranker (`make ablations`) | `scripts/ablations.py`, `docs/ablations.md` |
| 4.5 Containers and smoke test | **Built**: UI container, PDF export, one-command smoke test (`make smoke`) | `docker/Dockerfile.ui`, `app/ui/report_pdf.py`, `scripts/smoke_test.py` |

## Quick start
```bash
git clone <repo> && cd procurement-vendor-risk-copilot
make setup && source .venv/bin/activate          # or: pip install -r requirements.txt
cp .env.example .env                              # fill in endpoint + keys
# put the raw dataset at data/raw/procurement_dataset/  (shared via Drive; not in Git)
make corpus                                       # rebuild ground truth, policies, synthetic docs, corpus v1
make test                                         # 276 tests (offline)
make index STRATEGIES=fixed512_ctx,structure      # load the store (+ Gemini embeddings; resumable)
make api                                          # API on http://localhost:8000/docs (no Docker needed)
make ui                                           # Streamlit wireframes (mock JSON or live API)
make up                                           # Docker: Postgres+pgvector, Langfuse (:3000), API (:8000)
```
Requires Python 3.11+, `pdftotext` (poppler-utils) and Docker.

## Repository layout
```
app/
  api/main.py, repository.py               FastAPI routes, API-key auth, rate limit; reviews/turns/audit persistence
  tools/                                   five read-only tools + run_tool() (schemas, timeout, trace, audit)
  observability/tracing.py                 Langfuse + local JSONL traces for every tool and LLM call
  safety.py                                injection detection, PII masking, non-evidence document types
  ui/streamlit_app.py, mock_review.json    wireframes: case view, evidence, scorecard, approval
  agents/                                  Week 3 (empty)
  db/store.py                              document/chunk/embedding/vendor-record store: SQLite or Postgres (step 2.1)
  db/models.py, migrate.py                 SQLAlchemy tables for the API; run Alembic migrations (migrations/)
  db/schema.sql                            vendors, documents, chunks (pgvector + full-text), reviews,
                                           conversation_turns, financial_metrics, append-only audit_events
  scoring/rubric.yaml, engine.py           deterministic rubric (becomes the compliance / financial tools)
  schemas.py                               Pydantic models for vendor facts and every copilot output
  llm_client.py                            the only way to call the LLM: token ledger, quota guard, JSON retry, mock backend
  prompting.py, failure_analysis.py        step 1.6 strategies and step 1.7 failure classifier
  retrieval/chunking.py                    4 chunking strategies (fixed512, fixed512_ctx, structure, small2big)
  retrieval/retrievers.py                  BM25, dense, hybrid (RRF), reranker, filters, cited passages
  retrieval/gemini_embed.py                Gemini embeddings: task prompts, pacing, retries, resumable, query cache
  retrieval/evaluation.py, service.py      retrieval metrics (incl. quote@k); search() entry point for the tools
data/
  raw/            NOT IN GIT: original downloads (~260 MB)
  manifests/      source register, CUAD/NDA selections, synthetic doc list, corpus_v1_manifest (frozen)
  ground_truth/   35 vendors, expected outcomes, document plan
  seed/           rows for the vendors and financial_metrics tables
  synthetic/      444 generated docs: policy/ contracts/ compliance_evidence/ performance_purchase_records/ adversarial/
  processed/      NOT IN GIT: corpus_v1.jsonl (make ingest)
  evaluation/     eval_v1.jsonl (80 queries, frozen), sha256, README, CHANGELOG
docs/             spec, decision log, baselines, failure notes, threat model; experiments/ (chunking, retrieval)
scripts/          build_vendors, generate_policies, generate_synthetic_docs, ingest, retrieval_smoke_test,
                  build_eval_set, grade, retrieval_baseline, prompt_baseline, export_json_schemas,
                  index_corpus (2.1), chunking_experiments (2.2), retrieval_eval (2.4), check_gemini, make_ui_mock
tests/            ground truth, corpus, eval set, grader, baselines, retrieval, tools, API, safety, tracing, UI (243 tests)
reports/          baseline results (JSON); runs/ holds raw LLM outputs (not in Git)
docker/           docker-compose (db, langfuse, api), Dockerfile.api, api-entrypoint.sh
observability/    token ledger, traces.jsonl (not in Git); Langfuse setup in observability/README.md
```

## Data provenance
- **Real sources** (licences in `data/manifests/source_register.csv`):
  - Indian GFR 2017 and procurement manuals; DPDP Act 2023 and Rules 2025
  - UK Supplier Assurance Framework; World Bank procurement guidance
  - NIST SP 800-161r1; CSA CCM/CAIQ v4.1
  - CUAD, ContractNLI and ACORD; Common Paper standard agreements
- **Synthetic data:** the buyer (Indrapath Technologies Ltd), all 35 vendors, auditors, insurers, people and figures are fictional, and every vendor document is generated from `data/ground_truth/vendors.json`. Before any public demo, search each vendor name once to make sure it doesn't match a real company.

See `CONTRIBUTING.md` for branching, owners and rules.
