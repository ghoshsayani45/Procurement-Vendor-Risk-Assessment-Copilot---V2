#!/bin/sh
# Migrate the database, load corpus v1 into Postgres (idempotent: a restart rewrites nothing), then start the API.
set -e
python -m app.db.migrate
if [ -f data/processed/corpus_v1.jsonl ]; then
  python -m scripts.index_corpus --strategies "${RETRIEVAL_STRATEGY:-fixed512_ctx}" > /tmp/index_report.json \
    && echo "corpus loaded into $STORE_BACKEND store" || { echo "index_corpus failed:"; cat /tmp/index_report.json; exit 1; }
else
  echo "WARNING: data/processed/corpus_v1.jsonl not found (run 'make corpus' on the host); document_search will fail."
fi
exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --proxy-headers
