from __future__ import annotations
"""
MMR (Maximal Marginal Relevance): diverse retrieval that avoids near-duplicate results.
Trade-off: lambda=1.0 = pure relevance, lambda=0.0 = pure diversity.
"""
import os
import math
import httpx

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

DOCS = [
    {"id": "d01", "text": "Python try/except blocks catch exceptions. Use specific exception types like ValueError or FileNotFoundError."},
    {"id": "d02", "text": "Python exception handling: catch specific exceptions, avoid bare except, use finally for cleanup."},
    {"id": "d03", "text": "Handle errors in Python with try/except/finally. Log exceptions with logging.exception() for stack traces."},
    {"id": "d04", "text": "PostgreSQL error codes: 23505 unique violation, 23503 foreign key violation, 08006 connection failure."},
    {"id": "d05", "text": "Database connection errors in Python: catch psycopg2.OperationalError, use connection pooling to handle failures."},
    {"id": "d06", "text": "SQLAlchemy handles database errors: use Session.rollback() after exceptions, pool_pre_ping=True for dead connections."},
    {"id": "d07", "text": "Redis error handling: catch ConnectionError and TimeoutError, implement circuit breaker pattern for resilience."},
    {"id": "d08", "text": "Network timeout configuration: connect_timeout=5, read_timeout=30, use exponential backoff for retries."},
    {"id": "d09", "text": "Kubernetes pod crashloop: check logs with kubectl logs, common cause is missing env vars or bad DB URL."},
    {"id": "d10", "text": "Circuit breaker pattern: CLOSED → OPEN after N failures, HALF-OPEN to test recovery, prevents cascade failures."},
]

QUERY = "how to handle errors in Python"


def embed(text: str) -> list[float] | None:
    try:
        r = httpx.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30
        )
        return r.json()["embedding"]
    except Exception:
        return None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


def mmr(
    query_vec: list[float],
    doc_vecs: list[tuple[str, list[float]]],  # [(doc_id, vec)]
    top_k: int = 5,
    lambda_: float = 0.5,
) -> list[str]:
    """
    Iteratively selects the doc that maximises:
        lambda * sim(query, doc) - (1-lambda) * max(sim(selected, doc))

    lambda=1.0 → pure relevance (same as cosine ranking)
    lambda=0.0 → pure diversity (never pick similar docs)
    lambda=0.5 → balanced (default)
    """
    selected = []
    remaining = list(doc_vecs)

    for _ in range(min(top_k, len(remaining))):
        best_id, best_score = None, float("-inf")
        for doc_id, dvec in remaining:
            rel = cosine(query_vec, dvec)
            if selected:
                redundancy = max(cosine(svec, dvec) for _, svec in selected)
            else:
                redundancy = 0.0
            score = lambda_ * rel - (1 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_id = doc_id
                best_vec = dvec

        selected.append((best_id, best_vec))
        remaining = [(d, v) for d, v in remaining if d != best_id]

    return [doc_id for doc_id, _ in selected]


def greedy_top_k(query_vec: list[float], doc_vecs: list[tuple[str, list[float]]], top_k: int = 5) -> list[str]:
    ranked = sorted(doc_vecs, key=lambda x: cosine(query_vec, x[1]), reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MMR DIVERSITY DEMO ===\n")
    print(f"Query: {QUERY!r}\n")

    try:
        httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        ollama_ok = True
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        print("Ollama not running — showing MMR algorithm without live embeddings.\n")
        print("""
MMR Algorithm (pseudocode):
  selected = []
  remaining = all_documents

  for i in range(top_k):
      best = argmax over remaining:
          lambda * sim(query, doc) - (1-lambda) * max(sim(s, doc) for s in selected)
      selected.append(best)
      remaining.remove(best)

  return selected

Key insight: the second term penalises docs similar to ALREADY selected docs.
This prevents returning 5 near-identical results.

lambda values:
  1.0  →  pure relevance (same as cosine ranking)
  0.7  →  mostly relevant with some diversity
  0.5  →  balanced (typical production default)
  0.3  →  mostly diverse
  0.0  →  pure diversity (maximally spread docs regardless of query)
""")
        print("Why MMR matters:")
        print("  Without MMR: top-3 results for 'Python error handling' might all be about try/except")
        print("  With MMR:    top-3 covers try/except + database errors + logging — more useful")
    else:
        print("Embedding documents...")
        qvec = embed(QUERY)
        if not qvec:
            print("Embedding failed — check Ollama is running nomic-embed-text")
            raise SystemExit(1)

        doc_vecs = []
        for doc in DOCS:
            dvec = embed(doc["text"])
            if dvec:
                doc_vecs.append((doc["id"], dvec))

        doc_map = {d["id"]: d for d in DOCS}

        print(f"\n{'─'*70}")
        print("GREEDY TOP-5 (pure relevance, lambda=1.0):")
        print(f"{'─'*70}")
        greedy_ids = greedy_top_k(qvec, doc_vecs, top_k=5)
        for rank, doc_id in enumerate(greedy_ids, 1):
            sim = cosine(qvec, next(v for d, v in doc_vecs if d == doc_id))
            text = doc_map[doc_id]["text"][:70]
            print(f"  {rank}. [{doc_id}] sim={sim:.3f}  {text}...")

        print(f"\n{'─'*70}")
        for lam in [0.7, 0.5, 0.3]:
            print(f"MMR TOP-5 (lambda={lam}):")
            print(f"{'─'*70}")
            mmr_ids = mmr(qvec, doc_vecs, top_k=5, lambda_=lam)
            for rank, doc_id in enumerate(mmr_ids, 1):
                text = doc_map[doc_id]["text"][:70]
                print(f"  {rank}. [{doc_id}]  {text}...")
            print()

        print("Overlap between greedy and MMR(0.5):")
        mmr_ids_50 = mmr(qvec, doc_vecs, top_k=5, lambda_=0.5)
        overlap = set(greedy_ids) & set(mmr_ids_50)
        promoted = set(mmr_ids_50) - set(greedy_ids)
        print(f"  Shared: {sorted(overlap)}")
        print(f"  MMR promoted (not in greedy top-5): {sorted(promoted)}")
        print("\nKey takeaway: MMR surfaces docs from different semantic clusters.")
        print("Use lambda=0.5-0.7 in production RAG for better answer coverage.")
