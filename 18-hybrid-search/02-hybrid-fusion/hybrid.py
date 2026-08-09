"""
Hybrid search: BM25 + vector similarity fused with Reciprocal Rank Fusion (RRF).
A/B eval comparing pure BM25, pure vector, and hybrid approaches.
Requires: rank-bm25, Ollama at localhost:11434
"""
import os
import re
import math
import httpx
from collections import Counter

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except ImportError:
    BM25_OK = False


# ─── Corpus ──────────────────────────────────────────────────────────────────

DOCS = [
    {"id": "d01", "text": "PostgreSQL index optimization: use EXPLAIN ANALYZE to find slow queries, add indexes on WHERE clause columns, use partial indexes for filtered queries."},
    {"id": "d02", "text": "CustomerNotFoundException is thrown when customer_id lookup fails in CustomerRepository.findById(). Catch and return HTTP 404."},
    {"id": "d03", "text": "Exception handling best practices: catch specific exceptions, log with context, return meaningful error messages to callers."},
    {"id": "d04", "text": "Redis cache invalidation strategies: TTL-based expiry, write-through cache, cache-aside pattern, event-driven invalidation."},
    {"id": "d05", "text": "How to fix slow database queries: add indexes, use query planner hints, rewrite subqueries as CTEs, avoid N+1 query patterns."},
    {"id": "d06", "text": "QDRANT_API_KEY must be set as environment variable before starting the Qdrant server. Use os.environ.get('QDRANT_API_KEY')."},
    {"id": "d07", "text": "Kubernetes pod crash loop: check logs with kubectl logs, inspect events with kubectl describe pod, check resource limits."},
    {"id": "d08", "text": "Vector database performance tuning: set HNSW ef_construct=200 at index time, hnsw_ef=128 at query time for 99% recall."},
    {"id": "d09", "text": "Handling errors in Python: use try/except/finally, avoid bare except clauses, use logging.exception() to capture stack traces."},
    {"id": "d10", "text": "SQL injection prevention: always use parameterized queries, never concatenate user input into SQL strings, use an ORM."},
    {"id": "d11", "text": "CREATE EXTENSION vector; — this installs pgvector. Then: ALTER TABLE docs ADD COLUMN embedding vector(768);"},
    {"id": "d12", "text": "Monitoring SLOs: define error budget, track request latency p99, alert on burn rate exceeding 2x normal."},
]

# Test queries with known "best" answer for eval
EVAL_QUERIES = [
    {
        "query": "CustomerNotFoundException",
        "relevant": ["d02", "d03", "d09"],  # exact match first, then semantic
        "type": "exact",
    },
    {
        "query": "how to handle errors in my code",
        "relevant": ["d09", "d03", "d02"],
        "type": "semantic",
    },
    {
        "query": "EXPLAIN ANALYZE slow query",
        "relevant": ["d01", "d05"],
        "type": "mixed",
    },
    {
        "query": "database performance optimization",
        "relevant": ["d01", "d05", "d08"],
        "type": "semantic",
    },
    {
        "query": "QDRANT_API_KEY",
        "relevant": ["d06", "d08"],
        "type": "exact",
    },
    {
        "query": "exception handling best practices",
        "relevant": ["d09", "d03", "d02"],
        "type": "semantic",
    },
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    ma = math.sqrt(sum(x*x for x in a))
    mb = math.sqrt(sum(x*x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


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


def rrf(result_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top_k = set(retrieved[:k])
    return len(top_k & set(relevant)) / len(relevant)


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== HYBRID SEARCH DEMO ===\n")

texts = [d["text"] for d in DOCS]
doc_ids = [d["id"] for d in DOCS]

# Check Ollama
try:
    httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    OLLAMA_OK = True
except Exception:
    OLLAMA_OK = False

# BM25 setup
if BM25_OK:
    bm25 = BM25Okapi([tokenize(t) for t in texts])

def bm25_search(query: str, k: int = 5) -> list[str]:
    if not BM25_OK:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [doc_ids[i] for i in ranked[:k]]

# Vector embeddings (pre-computed)
doc_embeddings: dict[str, list[float]] = {}
if OLLAMA_OK:
    print("Pre-computing embeddings for corpus...")
    for doc in DOCS:
        emb = embed(doc["text"])
        if emb:
            doc_embeddings[doc["id"]] = emb
    print(f"  Embedded {len(doc_embeddings)} documents.\n")

def vector_search(query: str, k: int = 5) -> list[str]:
    if not doc_embeddings:
        return []
    qvec = embed(query)
    if not qvec:
        return []
    scored = [(doc_id, cosine(qvec, emb)) for doc_id, emb in doc_embeddings.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored[:k]]

def hybrid_search(query: str, k: int = 5) -> list[str]:
    bm = bm25_search(query, k=k*2)
    vec = vector_search(query, k=k*2)
    merged = rrf([bm, vec] if bm and vec else [bm or vec])
    return merged[:k]

# ─── Per-query comparison
print("─── Per-query results ───\n")
for q in EVAL_QUERIES:
    query = q["query"]
    relevant = q["relevant"]
    qtype = q["type"]

    bm_results = bm25_search(query) if BM25_OK else []
    vec_results = vector_search(query) if OLLAMA_OK else []
    hyb_results = hybrid_search(query) if (BM25_OK or OLLAMA_OK) else []

    bm_r  = recall_at_k(bm_results, relevant, 3)
    vec_r = recall_at_k(vec_results, relevant, 3)
    hyb_r = recall_at_k(hyb_results, relevant, 3)

    print(f"  Query: {query!r}  [{qtype}]")
    print(f"    BM25 top-3:   {bm_results[:3]}  R@3={bm_r:.2f}")
    print(f"    Vector top-3: {vec_results[:3]}  R@3={vec_r:.2f}")
    print(f"    Hybrid top-3: {hyb_results[:3]}  R@3={hyb_r:.2f}")
    winner = max(["BM25","Vector","Hybrid"], key=lambda n: {"BM25":bm_r,"Vector":vec_r,"Hybrid":hyb_r}[n])
    print(f"    Winner: {winner}\n")

# ─── Aggregate stats
if BM25_OK or OLLAMA_OK:
    print("─── Aggregate Recall@3 ───\n")
    bm_total = vec_total = hyb_total = 0.0
    for q in EVAL_QUERIES:
        bm_total  += recall_at_k(bm25_search(q["query"]) if BM25_OK else [], q["relevant"], 3)
        vec_total += recall_at_k(vector_search(q["query"]) if OLLAMA_OK else [], q["relevant"], 3)
        hyb_total += recall_at_k(hybrid_search(q["query"]) if (BM25_OK or OLLAMA_OK) else [], q["relevant"], 3)
    n = len(EVAL_QUERIES)
    print(f"  BM25:   {bm_total/n:.2f}")
    print(f"  Vector: {vec_total/n:.2f}")
    print(f"  Hybrid: {hyb_total/n:.2f}  ← typically wins")
    print()
    print("Hybrid wins because:")
    print("  - BM25 catches exact identifiers, code, error names")
    print("  - Vector catches semantic concepts even without exact words")
    print("  - RRF fusion rewards docs that appear in both ranked lists")
else:
    print("Install rank-bm25 and start Ollama to run the full eval.")
    print("  pip install rank-bm25")
    print("  ollama serve")
