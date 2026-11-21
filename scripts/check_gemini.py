"""Check the Gemini embeddings setup before a long indexing run.

    python -m scripts.check_gemini

Prints the settings in use (never the key itself), then sends 1 query and a batch of 20 short documents, and reports
vector size and latency. A failure is explained (key, model, quota, or a network/firewall reset).
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import find_dotenv, load_dotenv

path = find_dotenv(usecwd=True)
load_dotenv(path)


def main():
    from app.retrieval.gemini_embed import DEFAULT_BASE, GeminiEmbedder, GeminiError
    key = os.getenv("GEMINI_API_KEY", "").strip()
    print(f"{'.env file:':20s}{path or '(not found)'}")
    print(f"{'EMBED_BACKEND:':20s}{os.getenv('EMBED_BACKEND', 'gemini (default)')}")
    print(f"{'GEMINI_API_KEY:':20s}{'set, ' + str(len(key)) + ' characters' if key and key != 'REPLACE-ME' else 'MISSING'}")
    for k, d in (("GEMINI_EMBED_MODEL", "gemini-embedding-2"), ("GEMINI_EMBED_DIM", "768"), ("GEMINI_RPM", "100"),
                 ("GEMINI_TPM", "30000"), ("GEMINI_EMBED_BATCH", "100"), ("GEMINI_COUNT_ITEMS", "0"),
                 ("GEMINI_BASE_URL", DEFAULT_BASE), ("GEMINI_CA_BUNDLE", "")):
        print(f"{k + ':':20s}{os.getenv(k) if os.getenv(k) else (d or '(none)') + '  (default)'}")
    if not key or key == "REPLACE-ME":
        sys.exit("\nSet GEMINI_API_KEY in .env (https://aistudio.google.com/apikey).")
    try:
        g = GeminiEmbedder(ledger=False, max_retries=2)
        t0 = time.time()
        q = g._post([g._query_request("connection check")], 10)[0]     # direct call: not stored in the query cache
        t1 = time.time()
        d = g.embed_documents([("check", f"short test document number {i}") for i in range(20)])
        t2 = time.time()
    except GeminiError as e:
        sys.exit(f"\nFAILED: {e}")
    print(f"\nOK: query vector {len(q)} dims in {t1 - t0:.1f}s; batch of 20 documents -> {d.shape} in {t2 - t1:.1f}s")
    print("Ready for: python -m scripts.index_corpus --strategies fixed512_ctx,structure --embed")


if __name__ == "__main__":
    main()
