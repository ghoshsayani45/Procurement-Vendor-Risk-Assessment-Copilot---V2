-- Procurement & Vendor Risk Copilot schema (Week 2). Loaded by docker compose on first start, and re-applied by
-- app/db/store.py on connect (every statement is idempotent). After pulling a schema change: make down && docker volume rm pvrc_pgdata && make up
-- Tables follow the solution document, plus conversation_turns for multi-turn state.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vendors (
    id              TEXT PRIMARY KEY,           -- V001..V035
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('ACTIVE','PROSPECT','DEBARRED','EXITED')),
    owner           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,           -- corpus doc_id
    vendor_id       TEXT REFERENCES vendors(id),
    source_id       TEXT NOT NULL,              -- source register id (POL-001, SYN-..., CON-001 ...)
    slice           TEXT NOT NULL,
    type            TEXT NOT NULL,              -- doc_type = evidence type (soc2_report, contract, invoice ...)
    title           TEXT NOT NULL,
    version         TEXT,
    effective_date  DATE,
    status          TEXT NOT NULL DEFAULT 'CURRENT' CHECK (status IN ('CURRENT','SUPERSEDED')),
    access_class    TEXT NOT NULL DEFAULT 'internal' CHECK (access_class IN ('public','internal','restricted')),
    licence         TEXT,
    sha1            TEXT NOT NULL,              -- SHA-1 of the text, as in corpus_v1_manifest.csv (idempotent re-ingest)
    words           INT,
    corpus_version  TEXT NOT NULL
);

-- One row per chunk per chunking strategy (step 2.2 compares several). index_text = contextual header + text.
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT NOT NULL,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    strategy        TEXT NOT NULL,              -- fixed512 | fixed512_ctx | structure | small2big
    seq             INT  NOT NULL,
    parent_id       TEXT,                       -- small2big: the structure chunk returned in place of this child
    heading         TEXT,
    text            TEXT NOT NULL,
    index_text      TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    keywords        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', index_text)) STORED,
    PRIMARY KEY (id, strategy)
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (document_id, strategy);
CREATE INDEX IF NOT EXISTS chunks_fts ON chunks USING GIN (keywords);

-- Embeddings per (chunk, model), so the local model and the course endpoint can coexist. Exact search is fine at this size.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id        TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    dim             INT  NOT NULL,
    embedding       VECTOR NOT NULL,
    PRIMARY KEY (chunk_id, model_id)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id              BIGSERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    corpus_version  TEXT,
    chunker_version TEXT,
    strategy        TEXT,
    detail          JSONB
);

-- Vendor records: the procurement system of record (certificates on file, contract terms, financials, performance).
-- Loaded from data/ground_truth/vendors.json WITHOUT the answer-key fields (planted_issues, traps, adversarial, notes).
CREATE TABLE IF NOT EXISTS vendor_records (
    vendor_id       TEXT PRIMARY KEY REFERENCES vendors(id),
    record          JSONB NOT NULL,
    sha1            TEXT NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS financial_metrics (
    vendor_id       TEXT NOT NULL REFERENCES vendors(id),
    period          TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           NUMERIC NOT NULL,
    source          TEXT NOT NULL,
    PRIMARY KEY (vendor_id, period, metric)
);

CREATE TABLE IF NOT EXISTS reviews (
    id              UUID PRIMARY KEY,
    vendor_id       TEXT NOT NULL REFERENCES vendors(id),
    review_type     TEXT NOT NULL CHECK (review_type IN ('NEW_VENDOR','RENEWAL')),
    status          TEXT NOT NULL DEFAULT 'OPEN',
    as_of           DATE NOT NULL,
    score_json      JSONB,
    report_json     JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id              BIGSERIAL PRIMARY KEY,
    review_id       UUID NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant','tool')),
    content         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Append-only: who asked what, which tools ran, which sources, which human decisions.
CREATE TABLE IF NOT EXISTS audit_events (
    id              BIGSERIAL PRIMARY KEY,
    review_id       UUID REFERENCES reviews(id),
    actor           TEXT NOT NULL,              -- user id, 'copilot', or tool name
    action          TEXT NOT NULL,              -- REQUEST | TOOL_CALL | REFUSAL | INJECTION_DETECTED | HUMAN_DECISION ...
    decision        TEXT,
    detail          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE OR REPLACE RULE audit_no_update AS ON UPDATE TO audit_events DO INSTEAD NOTHING;
CREATE OR REPLACE RULE audit_no_delete AS ON DELETE TO audit_events DO INSTEAD NOTHING;
