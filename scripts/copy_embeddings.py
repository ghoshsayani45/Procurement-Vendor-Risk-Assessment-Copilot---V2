"""Copy stored chunk vectors from the laptop's SQLite store into the Docker Postgres store (no re-embedding).

    python -m scripts.copy_embeddings                       # fixed512_ctx, local_BAAI/bge-small-en-v1.5
    python -m scripts.copy_embeddings --strategy structure --model-id local_BAAI/bge-small-en-v1.5

The Docker API reads Postgres. Its chunks are there (the container ran index_corpus), but the 14,161 vectors from the
Week 2 local run live only in data/processed/store.sqlite. This copies them over for the chunks Postgres has, under the
same model id, so the container gets hybrid search with EMBED_BACKEND=endpoint + EMBED_REUSE_VECTORS_FROM. Postgres is
reached at DATABASE_URL (localhost:5432 from the laptop, published by docker-compose). Safe to re-run.
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from app.db.store import PostgresStore, SQLiteStore


def main(argv=None):
    load_dotenv(override=False)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default=os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx"))
    ap.add_argument("--model-id", default=os.getenv("EMBED_REUSE_VECTORS_FROM") or "local_BAAI/bge-small-en-v1.5")
    ap.add_argument("--sqlite", default=os.getenv("STORE_SQLITE_PATH") or None)
    ap.add_argument("--batch", type=int, default=1000)
    a = ap.parse_args(argv)
    src = SQLiteStore(a.sqlite) if a.sqlite else SQLiteStore()
    dst = PostgresStore()
    ids = [c.chunk_id for c in dst.load_chunks(a.strategy)]
    if not ids:
        raise SystemExit(f"Postgres has no '{a.strategy}' chunks: run index_corpus in the container first "
                         f"(docker compose ... exec api python -m scripts.index_corpus --strategies {a.strategy})")
    already = dst.load_embeddings(a.model_id, ids)
    todo = [i for i in ids if i not in already]
    have = src.load_embeddings(a.model_id, todo)
    for s in range(0, len(todo), a.batch):
        part = {i: have[i] for i in todo[s:s + a.batch] if i in have}
        if part:
            dst.put_embeddings(a.model_id, part)
    missing = len(todo) - len(have)
    print(f"{a.strategy} / {a.model_id}: Postgres chunks {len(ids):,}; already there {len(already):,}; copied {len(have):,}; "
          f"not in SQLite {missing:,}")
    if missing:
        print("Chunks without a vector differ between the two stores (re-chunked since?). Embed them in the container: "
              "index_corpus --embed with EMBED_BACKEND=endpoint.")


if __name__ == "__main__":
    main()
