"""Document / chunk store (step 2.1). One interface, two backends:

    STORE_BACKEND=sqlite    (default) a single file, data/processed/store.sqlite. No Docker needed.
    STORE_BACKEND=postgres  Postgres + pgvector from `make up` (POSTGRES_* in .env); schema in app/db/schema.sql.

The store is the system of record for vendors, documents, chunks (per chunking strategy) and embeddings (per model).
Retrieval loads chunks and vectors from it into an in-process index (BM25 + numpy): corpus v1 is ~25k chunks at most,
so exact search in memory is faster than a round trip and identical across backends (decision D19).

Idempotent and versioned:
  * documents are upserted by doc_id and compared by SHA-1: unchanged documents are skipped;
  * chunks are replaced per (document, strategy) only when the document changed or CHUNKER_VERSION changed;
  * embeddings are keyed by (chunk_id, model_id) and only missing ones are computed;
  * every run is recorded in ingest_runs (corpus version, chunker version, strategy, counts).
Re-running the pipeline on the same corpus therefore writes nothing and never duplicates rows.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from app.retrieval.chunking import ROOT, Chunk

SQLITE_PATH = ROOT / "data" / "processed" / "store.sqlite"

DOC_COLS = ("id", "vendor_id", "source_id", "slice", "type", "title", "version", "effective_date", "status", "access_class",
            "licence", "sha1", "words", "corpus_version")
CHUNK_COLS = ("id", "document_id", "strategy", "seq", "parent_id", "heading", "text", "index_text", "chunker_version")

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS vendors (id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, status TEXT NOT NULL,
                                    owner TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, vendor_id TEXT REFERENCES vendors(id), source_id TEXT NOT NULL,
    slice TEXT NOT NULL, type TEXT NOT NULL, title TEXT NOT NULL, version TEXT, effective_date TEXT,
    status TEXT NOT NULL DEFAULT 'CURRENT', access_class TEXT NOT NULL DEFAULT 'internal', licence TEXT, sha1 TEXT NOT NULL,
    words INTEGER, corpus_version TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (id TEXT NOT NULL, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    strategy TEXT NOT NULL, seq INTEGER NOT NULL, parent_id TEXT, heading TEXT, text TEXT NOT NULL, index_text TEXT NOT NULL,
    chunker_version TEXT NOT NULL, PRIMARY KEY (id, strategy));
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (document_id, strategy);
CREATE TABLE IF NOT EXISTS chunk_embeddings (chunk_id TEXT NOT NULL, model_id TEXT NOT NULL, dim INTEGER NOT NULL,
    embedding BLOB NOT NULL, PRIMARY KEY (chunk_id, model_id));
CREATE TABLE IF NOT EXISTS ingest_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, corpus_version TEXT,
    chunker_version TEXT, strategy TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS vendor_records (vendor_id TEXT PRIMARY KEY REFERENCES vendors(id), record TEXT NOT NULL,
    sha1 TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


def _canon(record: dict) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)


def _sha(record: dict) -> str:
    import hashlib
    return hashlib.sha1(_canon(record).encode()).hexdigest()


def _doc_row(d: dict, corpus_version: str) -> dict:
    return dict(id=d["doc_id"], vendor_id=d.get("vendor_id"), source_id=d.get("source_id", ""), slice=d["slice"],
                type=d.get("doc_type", ""), title=d["title"], version=d.get("version"), effective_date=d.get("document_date"),
                status=d.get("status", "CURRENT"), access_class=d.get("access_class", "internal"), licence=d.get("licence"),
                sha1=d["sha1"], words=d.get("words"), corpus_version=corpus_version)


def _chunk_from_row(r: dict) -> Chunk:
    return Chunk(chunk_id=r["id"], doc_id=r["document_id"], seq=r["seq"], text=r["text"], vendor_id=r["vendor_id"],
                 slice=r["slice"], title=r["title"], strategy=r["strategy"], heading=r["heading"] or "",
                 parent_id=r["parent_id"], doc_type=r["type"], access_class=r["access_class"], status=r["status"],
                 document_date=str(r["effective_date"]) if r["effective_date"] else None, source_id=r["source_id"],
                 index_text=r["index_text"])


LOAD_SQL = """SELECT c.id, c.document_id, c.strategy, c.seq, c.parent_id, c.heading, c.text, c.index_text,
                     d.vendor_id, d.slice, d.title, d.type, d.access_class, d.status, d.effective_date, d.source_id
              FROM chunks c JOIN documents d ON d.id = c.document_id
              WHERE c.strategy = {p} ORDER BY c.document_id, c.seq"""


class SQLiteStore:
    backend = "sqlite"

    def __init__(self, path: Path | str = SQLITE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)   # tools run in worker threads; sqlite3 serialises access
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SQLITE_SCHEMA)

    # --- writes
    def upsert_vendors(self, rows: list[dict]) -> int:
        with self.conn:
            self.conn.executemany("""INSERT INTO vendors (id, name, category, status, owner) VALUES (:id, :name, :category, :status, :owner)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name, category=excluded.category, status=excluded.status,
                owner=excluded.owner""", rows)
        return len(rows)

    def document_hashes(self) -> dict[str, str]:
        return {r["id"]: r["sha1"] for r in self.conn.execute("SELECT id, sha1 FROM documents")}

    def upsert_documents(self, docs: list[dict], corpus_version: str) -> None:
        rows = [_doc_row(d, corpus_version) for d in docs]
        cols = ", ".join(DOC_COLS)
        sets = ", ".join(f"{c}=excluded.{c}" for c in DOC_COLS if c != "id")
        with self.conn:
            self.conn.executemany(f"INSERT INTO documents ({cols}) VALUES ({', '.join(':' + c for c in DOC_COLS)}) "
                                  f"ON CONFLICT(id) DO UPDATE SET {sets}", rows)

    def delete_documents(self, doc_ids: list[str]) -> None:
        with self.conn:
            self.conn.executemany("DELETE FROM chunks WHERE document_id = ?", [(d,) for d in doc_ids])
            self.conn.executemany("DELETE FROM documents WHERE id = ?", [(d,) for d in doc_ids])

    def chunk_versions(self, strategy: str) -> dict[str, str]:
        """document_id -> chunker_version of its chunks for this strategy (documents never chunked are absent)."""
        return {r[0]: r[1] for r in self.conn.execute(
            "SELECT document_id, MIN(chunker_version) FROM chunks WHERE strategy = ? GROUP BY document_id", (strategy,))}

    def replace_chunks(self, strategy: str, doc_ids: list[str], chunks: list[Chunk], chunker_version: str) -> None:
        with self.conn:
            self.conn.executemany("DELETE FROM chunks WHERE document_id = ? AND strategy = ?", [(d, strategy) for d in doc_ids])
            self.conn.executemany(f"INSERT INTO chunks ({', '.join(CHUNK_COLS)}) VALUES ({', '.join('?' * len(CHUNK_COLS))})",
                                  [(c.chunk_id, c.doc_id, strategy, c.seq, c.parent_id, c.heading, c.text, c.index_text,
                                    chunker_version) for c in chunks])

    def put_embeddings(self, model_id: str, vecs: dict[str, np.ndarray]) -> None:
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO chunk_embeddings (chunk_id, model_id, dim, embedding) VALUES (?, ?, ?, ?)",
                                  [(cid, model_id, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
                                   for cid, v in vecs.items()])

    def record_run(self, corpus_version: str, chunker_version: str, strategy: str, detail: dict) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO ingest_runs (started_at, corpus_version, chunker_version, strategy, detail) "
                              "VALUES (?, ?, ?, ?, ?)", (time.strftime("%Y-%m-%d %H:%M:%S"), corpus_version, chunker_version,
                                                         strategy, json.dumps(detail)))

    def upsert_vendor_records(self, records: list[dict]) -> int:
        """Vendor records (the procurement system of record). Returns how many were new or changed."""
        old = {r[0]: r[1] for r in self.conn.execute("SELECT vendor_id, sha1 FROM vendor_records")}
        rows = [(r["vendor_id"], _canon(r), _sha(r)) for r in records]
        todo = [x for x in rows if old.get(x[0]) != x[2]]
        with self.conn:
            self.conn.executemany("INSERT INTO vendor_records (vendor_id, record, sha1, updated_at) VALUES (?, ?, ?, ?) "
                                  "ON CONFLICT(vendor_id) DO UPDATE SET record=excluded.record, sha1=excluded.sha1, "
                                  "updated_at=excluded.updated_at", [(*x, time.strftime("%Y-%m-%d %H:%M:%S")) for x in todo])
        return len(todo)

    def get_vendor_records(self, vendor_ids: list[str] | None = None) -> dict[str, dict]:
        rows = self.conn.execute("SELECT vendor_id, record FROM vendor_records ORDER BY vendor_id").fetchall()
        return {r[0]: json.loads(r[1]) for r in rows if vendor_ids is None or r[0] in vendor_ids}

    def vendor_statuses(self) -> dict[str, str]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT id, status FROM vendors")}

    def vendor_texts(self, vendor_id: str) -> list[dict]:
        """Full text of each of the vendor's documents, rebuilt from one chunk set (for integrity scans)."""
        strat = self.conn.execute("SELECT strategy FROM chunks GROUP BY strategy ORDER BY COUNT(*) LIMIT 1").fetchone()
        if not strat:
            return []
        rows = self.conn.execute("SELECT d.id, d.title, d.type, c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
                                 "WHERE d.vendor_id = ? AND c.strategy = ? ORDER BY d.id, c.seq", (vendor_id, strat[0])).fetchall()
        out: dict[str, dict] = {}
        for doc_id, title, typ, text in rows:
            out.setdefault(doc_id, dict(doc_id=doc_id, title=title, doc_type=typ, text=""))["text"] += "\n" + text
        return list(out.values())

    def vendor_documents(self, vendor_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT id, type, title, effective_date, status FROM documents WHERE vendor_id = ? ORDER BY id",
                                 (vendor_id,)).fetchall()
        return [dict(doc_id=r[0], doc_type=r[1], title=r[2], document_date=r[3], status=r[4]) for r in rows]

    # --- reads
    def load_chunks(self, strategy: str) -> list[Chunk]:
        return [_chunk_from_row(dict(r)) for r in self.conn.execute(LOAD_SQL.format(p="?"), (strategy,))]

    def load_embeddings(self, model_id: str, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        want, out = set(chunk_ids), {}
        for cid, blob in self.conn.execute("SELECT chunk_id, embedding FROM chunk_embeddings WHERE model_id = ?", (model_id,)):
            if cid in want:
                out[cid] = np.frombuffer(blob, dtype=np.float32)
        return out

    def counts(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchall()
        return {"vendors": q("SELECT COUNT(*) FROM vendors")[0][0], "documents": q("SELECT COUNT(*) FROM documents")[0][0],
                "vendor_records": q("SELECT COUNT(*) FROM vendor_records")[0][0],
                "chunks": {r[0]: r[1] for r in q("SELECT strategy, COUNT(*) FROM chunks GROUP BY strategy")},
                "embeddings": {r[0]: r[1] for r in q("SELECT model_id, COUNT(*) FROM chunk_embeddings GROUP BY model_id")},
                "runs": q("SELECT COUNT(*) FROM ingest_runs")[0][0]}

    def close(self):
        self.conn.close()


class PostgresStore:
    """Same interface on Postgres + pgvector (docker/docker-compose.yml). Embeddings use the pgvector `vector` type."""
    backend = "postgres"

    def __init__(self, dsn: str | None = None):
        import psycopg
        from psycopg.rows import dict_row
        dsn = dsn or os.getenv("DATABASE_URL") or (
            f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.environ['POSTGRES_DB']}")
        self.conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)
        schema = (ROOT / "app" / "db" / "schema.sql").read_text(encoding="utf-8")
        with self.conn.cursor() as cur:
            cur.execute(schema)               # every statement is IF NOT EXISTS / OR REPLACE, so this is safe to repeat
        self.conn.commit()
        from pgvector.psycopg import register_vector
        register_vector(self.conn)

    def _many(self, sql, rows):
        with self.conn.cursor() as cur:
            cur.executemany(sql, rows)
        self.conn.commit()

    def upsert_vendors(self, rows: list[dict]) -> int:
        self._many("""INSERT INTO vendors (id, name, category, status, owner) VALUES (%(id)s, %(name)s, %(category)s, %(status)s, %(owner)s)
            ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, category=EXCLUDED.category, status=EXCLUDED.status,
            owner=EXCLUDED.owner""", rows)
        return len(rows)

    def document_hashes(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, sha1 FROM documents")
            return {r["id"]: r["sha1"] for r in cur.fetchall()}

    def upsert_documents(self, docs: list[dict], corpus_version: str) -> None:
        rows = [_doc_row(d, corpus_version) for d in docs]
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in DOC_COLS if c != "id")
        self._many(f"INSERT INTO documents ({', '.join(DOC_COLS)}) VALUES ({', '.join(f'%({c})s' for c in DOC_COLS)}) "
                   f"ON CONFLICT (id) DO UPDATE SET {sets}", rows)

    def delete_documents(self, doc_ids: list[str]) -> None:
        self._many("DELETE FROM documents WHERE id = %s", [(d,) for d in doc_ids])   # chunks cascade

    def chunk_versions(self, strategy: str) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT document_id, MIN(chunker_version) AS v FROM chunks WHERE strategy = %s GROUP BY document_id",
                        (strategy,))
            return {r["document_id"]: r["v"] for r in cur.fetchall()}

    def replace_chunks(self, strategy: str, doc_ids: list[str], chunks: list[Chunk], chunker_version: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE strategy = %s AND document_id = ANY(%s)", (strategy, doc_ids))
            cur.executemany(f"INSERT INTO chunks ({', '.join(CHUNK_COLS)}) VALUES ({', '.join(['%s'] * len(CHUNK_COLS))})",
                            [(c.chunk_id, c.doc_id, strategy, c.seq, c.parent_id, c.heading, c.text, c.index_text,
                              chunker_version) for c in chunks])
        self.conn.commit()

    def put_embeddings(self, model_id: str, vecs: dict[str, np.ndarray]) -> None:
        self._many("""INSERT INTO chunk_embeddings (chunk_id, model_id, dim, embedding) VALUES (%s, %s, %s, %s)
                      ON CONFLICT (chunk_id, model_id) DO UPDATE SET embedding = EXCLUDED.embedding, dim = EXCLUDED.dim""",
                   [(cid, model_id, int(v.shape[0]), np.asarray(v, dtype=np.float32)) for cid, v in vecs.items()])

    def record_run(self, corpus_version: str, chunker_version: str, strategy: str, detail: dict) -> None:
        self._many("INSERT INTO ingest_runs (corpus_version, chunker_version, strategy, detail) VALUES (%s, %s, %s, %s)",
                   [(corpus_version, chunker_version, strategy, json.dumps(detail))])

    def upsert_vendor_records(self, records: list[dict]) -> int:
        old = {r["vendor_id"]: r["sha1"] for r in self._fetch("SELECT vendor_id, sha1 FROM vendor_records")}
        todo = [(r["vendor_id"], _canon(r), _sha(r)) for r in records if old.get(r["vendor_id"]) != _sha(r)]
        self._many("""INSERT INTO vendor_records (vendor_id, record, sha1) VALUES (%s, %s::jsonb, %s)
                      ON CONFLICT (vendor_id) DO UPDATE SET record = EXCLUDED.record, sha1 = EXCLUDED.sha1, updated_at = now()""", todo)
        return len(todo)

    def get_vendor_records(self, vendor_ids: list[str] | None = None) -> dict[str, dict]:
        rows = self._fetch("SELECT vendor_id, record FROM vendor_records ORDER BY vendor_id")
        return {r["vendor_id"]: (r["record"] if isinstance(r["record"], dict) else json.loads(r["record"]))
                for r in rows if vendor_ids is None or r["vendor_id"] in vendor_ids}

    def vendor_statuses(self) -> dict[str, str]:
        return {r["id"]: r["status"] for r in self._fetch("SELECT id, status FROM vendors")}

    def vendor_texts(self, vendor_id: str) -> list[dict]:
        strat = self._fetch("SELECT strategy FROM chunks GROUP BY strategy ORDER BY COUNT(*) LIMIT 1")
        if not strat:
            return []
        rows = self._fetch("SELECT d.id, d.title, d.type, c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
                           "WHERE d.vendor_id = %s AND c.strategy = %s ORDER BY d.id, c.seq", (vendor_id, strat[0]["strategy"]))
        out: dict[str, dict] = {}
        for r in rows:
            out.setdefault(r["id"], dict(doc_id=r["id"], title=r["title"], doc_type=r["type"], text=""))["text"] += "\n" + r["text"]
        return list(out.values())

    def vendor_documents(self, vendor_id: str) -> list[dict]:
        rows = self._fetch("SELECT id, type, title, effective_date, status FROM documents WHERE vendor_id = %s ORDER BY id", (vendor_id,))
        return [dict(doc_id=r["id"], doc_type=r["type"], title=r["title"],
                     document_date=str(r["effective_date"]) if r["effective_date"] else None, status=r["status"]) for r in rows]

    def _fetch(self, sql, params=()):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        self.conn.commit()
        return rows

    def load_chunks(self, strategy: str) -> list[Chunk]:
        with self.conn.cursor() as cur:
            cur.execute(LOAD_SQL.format(p="%s"), (strategy,))
            return [_chunk_from_row(r) for r in cur.fetchall()]

    def load_embeddings(self, model_id: str, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, embedding FROM chunk_embeddings WHERE model_id = %s AND chunk_id = ANY(%s)",
                        (model_id, list(chunk_ids)))
            return {r["chunk_id"]: np.asarray(r["embedding"], dtype=np.float32) for r in cur.fetchall()}

    def vector_search(self, model_id: str, query_vec: np.ndarray, strategy: str, k: int = 20) -> list[tuple[str, float]]:
        """Exact cosine search inside Postgres (pgvector `<=>`), for checking the in-process index or larger corpora."""
        with self.conn.cursor() as cur:
            cur.execute("""SELECT e.chunk_id, 1 - (e.embedding <=> %s) AS score FROM chunk_embeddings e
                           JOIN chunks c ON c.id = e.chunk_id AND c.strategy = %s
                           WHERE e.model_id = %s ORDER BY e.embedding <=> %s LIMIT %s""",
                        (np.asarray(query_vec, dtype=np.float32), strategy, model_id, np.asarray(query_vec, dtype=np.float32), k))
            return [(r["chunk_id"], float(r["score"])) for r in cur.fetchall()]

    def counts(self) -> dict:
        with self.conn.cursor() as cur:
            def one(sql):
                cur.execute(sql)
                return cur.fetchall()
            return {"vendors": one("SELECT COUNT(*) AS n FROM vendors")[0]["n"],
                    "vendor_records": one("SELECT COUNT(*) AS n FROM vendor_records")[0]["n"],
                    "documents": one("SELECT COUNT(*) AS n FROM documents")[0]["n"],
                    "chunks": {r["strategy"]: r["n"] for r in one("SELECT strategy, COUNT(*) AS n FROM chunks GROUP BY strategy")},
                    "embeddings": {r["model_id"]: r["n"] for r in one("SELECT model_id, COUNT(*) AS n FROM chunk_embeddings GROUP BY model_id")},
                    "runs": one("SELECT COUNT(*) AS n FROM ingest_runs")[0]["n"]}

    def close(self):
        self.conn.close()


def get_store(backend: str | None = None):
    backend = (backend or os.getenv("STORE_BACKEND", "sqlite")).strip().lower()
    if backend == "sqlite":
        return SQLiteStore(os.getenv("STORE_SQLITE_PATH", SQLITE_PATH))
    if backend in ("postgres", "pg"):
        return PostgresStore()
    raise ValueError(f"STORE_BACKEND={backend!r}: use sqlite or postgres")
