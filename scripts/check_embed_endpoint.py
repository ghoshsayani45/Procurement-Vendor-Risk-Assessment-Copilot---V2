"""Check the course embeddings endpoint, and whether its vectors match the ones already in your store.

    python -m scripts.check_embed_endpoint
    python -m scripts.check_embed_endpoint --sample 40 --compare-with local_BAAI/bge-small-en-v1.5

  1. settings    EMBED_BASE_URL, EMBED_MODEL, key length (never the key), EMBED_CA_BUNDLE exists
  2. call        embed two texts: dimension, latency, normalised?
  3. same space  embed --sample stored chunks again through the endpoint and compare with their stored vectors
                 (cosine). The endpoint serves bge-small-en-v1.5, the model the Week 2 local run used: if the vectors
                 agree, the 14,161 stored vectors can be reused instead of re-embedding everything.
  4. verdict     the .env lines to use

Costs a few thousand embedding tokens. Reads .env; never prints the key.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
from dotenv import load_dotenv

REUSE_MIN_MEAN, REUSE_MIN_WORST = 0.995, 0.98


def say(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def explain(e: Exception) -> str:
    t = f"{type(e).__name__}: {e}"
    low = t.lower()
    if "certificate" in low or "ssl" in low:
        return (t[:300] + "\n       -> TLS: set EMBED_CA_BUNDLE to the course certificate (e.g. C:/Users/<you>/Downloads/vm02-ca.crt, "
                "forward slashes). Never turn verification off.")
    if "401" in low or "403" in low or "unauthorized" in low or "authentication" in low:
        return t[:300] + "\n       -> key rejected: check EMBED_API_KEY (the shared embedding key, not the chat key)"
    if "404" in low:
        return t[:300] + "\n       -> wrong URL or model: EMBED_BASE_URL must end in /v1; EMBED_MODEL=bge-small-en-v1.5"
    if "connect" in low or "timed out" in low or "timeout" in low:
        return (t[:300] + "\n       -> cannot reach the server: check the URL and port, and whether your network/VPN allows it "
                "(an office proxy may need NO_PROXY=103.42.50.41)")
    return t[:400]


def main(argv=None) -> int:
    load_dotenv(override=False)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--strategy", default=os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx"))
    ap.add_argument("--compare-with", default="local_BAAI/bge-small-en-v1.5", help="model id of the stored vectors")
    a = ap.parse_args(argv)

    print("Embeddings endpoint check\n")
    url, model, key = os.getenv("EMBED_BASE_URL", ""), os.getenv("EMBED_MODEL", ""), os.getenv("EMBED_API_KEY", "")
    ca = os.getenv("EMBED_CA_BUNDLE", "") or os.getenv("LLM_CA_BUNDLE", "")
    say("    ", f"EMBED_BASE_URL = {url or '(empty)'}; EMBED_MODEL = {model or '(empty)'}; EMBED_API_KEY: {len(key)} chars"
               + ("" if key else " (empty: LLM_API_KEY would be used)"))
    say("    ", f"EMBED_CA_BUNDLE = {ca or '(empty: the OS certificate store is used)'}")
    bad = []
    if not url or "REPLACE" in url:
        bad.append("EMBED_BASE_URL is not set (e.g. https://103.42.50.41:1035/v1)")
    elif not url.rstrip("/").endswith("/v1"):
        bad.append("EMBED_BASE_URL should end in /v1 (the client adds /embeddings)")
    if not model or "REPLACE" in model:
        bad.append("EMBED_MODEL is not set (bge-small-en-v1.5)")
    if not key or "REPLACE" in key:
        bad.append("EMBED_API_KEY is empty: put the shared embedding key in .env")
    if ca and not os.path.isfile(os.path.expanduser(ca.strip('"'))):
        bad.append(f"EMBED_CA_BUNDLE file not found: {ca}")
    if bad:
        for b in bad:
            say("FAIL", b)
        return 1
    say("OK  ", "settings look right")

    from app.llm_client import embed
    try:
        t0 = time.time()
        v = np.asarray(embed(["vendor risk assessment", "SOC 2 Type II report expired"], tag="check"), dtype=np.float32)
        ms = (time.time() - t0) * 1000
    except Exception as e:
        say("FAIL", explain(e))
        return 1
    norms = np.linalg.norm(v, axis=1)
    say("OK  ", f"endpoint works: {v.shape[0]} vectors of {v.shape[1]} dims in {ms:.0f} ms; norms {norms.round(3).tolist()}"
                + (" (normalised)" if np.allclose(norms, 1, atol=1e-2) else " (not normalised: the client normalises them)"))
    if v.shape[1] != 384:
        say("WARN", f"bge-small-en-v1.5 has 384 dims, got {v.shape[1]}: this is not the model the local run used")

    from app.db.store import get_store
    store = get_store()
    chunks = store.load_chunks(a.strategy)
    have = store.load_embeddings(a.compare_with, [c.chunk_id for c in chunks]) if chunks else {}
    if not have:
        say("    ", f"no stored vectors for {a.compare_with} on '{a.strategy}' chunks: nothing to compare. Embed through the "
                   f"endpoint:\n       EMBED_BACKEND=endpoint, then python -m scripts.index_corpus --strategies {a.strategy} --embed")
        return 0
    random.seed(7)
    pick = random.sample([c for c in chunks if c.chunk_id in have], min(a.sample, len(have)))
    try:
        t0 = time.time()
        new = np.asarray(embed([c.index_text for c in pick], tag="check"), dtype=np.float32)
        reqs = -(-len(pick) // max(1, int(os.getenv("EMBED_MAX_BATCH", "16"))))
        say("    ", f"{len(pick)} chunks embedded in {(time.time() - t0) * 1000:.0f} ms (~{(time.time() - t0) * 1000 / reqs:.0f} ms "
                   f"per request of up to 16; long chunks clipped to 512 tokens like the local run)")
        t0 = time.time()
        embed(["what are the payment terms in the contract?"], tag="check")
        say("    ", f"one search query: {(time.time() - t0) * 1000:.0f} ms (paid on every document_search)")
    except Exception as e:
        say("FAIL", explain(e))
        return 1
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    old = np.stack([have[c.chunk_id] for c in pick]).astype(np.float32)
    old /= np.linalg.norm(old, axis=1, keepdims=True)
    cos = (new * old).sum(axis=1)
    say("    ", f"{len(pick)} stored chunks re-embedded: cosine to the stored vectors mean {cos.mean():.4f}, worst {cos.min():.4f}")

    print()
    if cos.mean() >= REUSE_MIN_MEAN and cos.min() >= REUSE_MIN_WORST:
        say("OK  ", "same vector space: reuse the stored vectors. In .env:")
        print(f"         EMBED_BACKEND=endpoint\n         EMBED_REUSE_VECTORS_FROM={a.compare_with}")
        print("       Queries go to the endpoint; the stored chunk vectors are used as they are (no 3-hour re-embed).")
    else:
        say("FAIL", f"the vectors differ (mean {cos.mean():.3f} < {REUSE_MIN_MEAN} or worst {cos.min():.3f} < {REUSE_MIN_WORST}): "
                    "different model, version or text handling. Do not mix them. Either keep EMBED_BACKEND=local, or "
                    f"re-embed through the endpoint: EMBED_BACKEND=endpoint (no EMBED_REUSE_VECTORS_FROM), then\n"
                    f"         python -m scripts.index_corpus --strategies {a.strategy} --embed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
