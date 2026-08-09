"""
Cross-encoder reranking: improve RAG precision by reranking bi-encoder candidates.
Compares bi-encoder ranking vs cross-encoder ranking on the same query.
Requires: sentence-transformers, Ollama for bi-encoder embeddings
"""
import os
import math
import httpx

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

try:
    from sentence_transformers import CrossEncoder
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False
    print("sentence-transformers not installed. pip install sentence-transformers\n")

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── Corpus ──────────────────────────────────────────────────────────────────

DOCS = [
    {"id": "d01", "title": "Handling Database Errors in Python",
     "text": "Use try/except blocks to catch psycopg2.OperationalError. Always close connections in finally blocks. Use context managers for automatic cleanup."},
    {"id": "d02", "title": "PostgreSQL Error Codes",
     "text": "PostgreSQL returns SQLSTATE codes. 23505 is unique violation. 23503 is foreign key violation. 08006 is connection failure. Use pg_catalog.pg_stat_activity to debug locks."},
    {"id": "d03", "title": "Python Exception Best Practices",
     "text": "Catch specific exceptions, not bare except. Use logging.exception() to capture stack traces. Raise from original exception to preserve context."},
    {"id": "d04", "title": "Database Connection Pooling",
     "text": "SQLAlchemy connection pools prevent connection exhaustion. Use pool_size=5, max_overflow=10. Pool pre_ping=True checks connections are alive before use."},
    {"id": "d05", "title": "Redis Error Handling",
     "text": "redis.ConnectionError and redis.TimeoutError are the two main failure modes. Use retry logic and circuit breakers for Redis cache failures."},
    {"id": "d06", "title": "Kubernetes Pod Crash Loop",
     "text": "CrashLoopBackOff means the container keeps failing. Check logs with kubectl logs. Common causes: missing env vars, bad database URL, port conflicts."},
    {"id": "d07", "title": "SQL Transaction Management",
     "text": "Use BEGIN/COMMIT/ROLLBACK for atomic operations. Savepoints allow partial rollback. Long transactions cause lock contention. Deadlock detection via pg_locks."},
    {"id": "d08", "title": "Python Logging Setup",
     "text": "Configure structlog or stdlib logging at app startup. Use log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL. Include request_id in every log entry for tracing."},
    {"id": "d09", "title": "Database Migration Errors",
     "text": "Alembic migration failures leave the database in unknown state. Always run migrations in a transaction. Use --sql flag to preview before applying."},
    {"id": "d10", "title": "Network Timeout Configuration",
     "text": "Set connect_timeout=5, read_timeout=30 for database connections. Use circuit breakers to fail fast when downstream services are down."},
]

QUERY = "how to handle errors when database connection fails"


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


def bi_encoder_search(query: str, docs: list[dict], top_k: int = 10) -> list[dict]:
    qvec = embed(query)
    if not qvec:
        return docs[:top_k]
    results = []
    for doc in docs:
        dvec = embed(doc["text"])
        if dvec:
            results.append({**doc, "score": cosine(qvec, dvec)})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def cross_encoder_rerank(query: str, candidates: list[dict], model_name: str, top_k: int = 5) -> list[dict]:
    model = CrossEncoder(model_name)
    pairs = [(query, doc["text"]) for doc in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        [({**doc, "ce_score": float(s)}) for doc, s in zip(candidates, scores)],
        key=lambda x: x["ce_score"],
        reverse=True
    )
    return ranked[:top_k]


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== CROSS-ENCODER RERANKING DEMO ===\n")
print(f"Query: {QUERY!r}\n")

if not CE_AVAILABLE:
    print("Showing reranking concepts (sentence-transformers not installed):\n")
    print("""
from sentence_transformers import CrossEncoder

# Load cross-encoder model (downloads ~80MB first time)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score query-document pairs together
pairs = [(query, doc) for doc in candidate_texts]
scores = model.predict(pairs)  # list of relevance scores

# Sort by score
ranked = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)
for text, score in ranked[:5]:
    print(f"  {score:.3f}  {text[:60]}")

# Key: scores are NOT cosine similarity (0-1)
# Cross-encoder scores are raw logits — relative ranking matters, not absolute value
# Higher = more relevant to query
""")
    print("Why two-stage retrieval wins:")
    print("  Bi-encoder: fast, handles 1M docs, ~85% precision at top-5")
    print("  Cross-encoder: slow (N forward passes), handles top-50, ~97% precision at top-5")
    print("  Combined: bi-encoder narrows to 50, cross-encoder finds true top-5")
    print("\nRecommended models (free, local):")
    print("  cross-encoder/ms-marco-MiniLM-L-6-v2  ~80MB  fastest, good quality")
    print("  cross-encoder/ms-marco-MiniLM-L-12-v2 ~130MB slower, better")
    print("  BAAI/bge-reranker-base                 multilingual, strong")
else:
    # Stage 1: bi-encoder retrieval
    try:
        httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        ollama_ok = True
    except Exception:
        ollama_ok = False

    if ollama_ok:
        print("Stage 1: Bi-encoder retrieval (top-10)...\n")
        stage1 = bi_encoder_search(QUERY, DOCS, top_k=10)
        print(f"  {'Rank':<5} {'Score':<8} Title")
        for i, doc in enumerate(stage1, 1):
            print(f"  {i:<5} {doc['score']:.3f}  {doc['title']}")
        print()
    else:
        print("Ollama not running — using all docs as stage-1 candidates.\n")
        stage1 = [{**doc, "score": 0.5} for doc in DOCS]

    # Stage 2: cross-encoder reranking
    print(f"Stage 2: Cross-encoder reranking with {CROSS_ENCODER_MODEL}...")
    print("  (downloading model on first run ~80MB)\n")
    stage2 = cross_encoder_rerank(QUERY, stage1, CROSS_ENCODER_MODEL, top_k=5)

    print(f"  {'Rank':<5} {'CE Score':<12} Title")
    for i, doc in enumerate(stage2, 1):
        print(f"  {i:<5} {doc['ce_score']:.4f}  {doc['title']}")
    print()

    if ollama_ok:
        # Compare ranking changes
        stage1_ids = [d["id"] for d in stage1[:5]]
        stage2_ids = [d["id"] for d in stage2]
        print("Ranking comparison (top-5):")
        print(f"  Bi-encoder:    {stage1_ids}")
        print(f"  Cross-encoder: {stage2_ids}")
        moved = [d for d in stage2_ids if d not in stage1_ids]
        if moved:
            print(f"  Reranker promoted: {moved} (weren't in bi-encoder top-5)")
        print()
        print("Key insight: cross-encoder re-orders results significantly.")
        print("The #1 bi-encoder result is often NOT the most relevant to the query.")
