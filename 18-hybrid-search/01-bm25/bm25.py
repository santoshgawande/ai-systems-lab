"""
BM25 full-text search: index documents, score queries, compare against naive TF-IDF.
Shows where BM25 beats vector search — exact terms, rare identifiers, code.
Requires: rank-bm25
"""
import math
import re
from collections import Counter

try:
    from rank_bm25 import BM25Okapi, BM25Plus
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("rank-bm25 not installed. pip install rank-bm25\n")


# ─── Corpus ──────────────────────────────────────────────────────────────────

DOCS = [
    {"id": 1,  "title": "PostgreSQL Index Types", "body": "PostgreSQL supports B-tree, Hash, GiST, SP-GiST, GIN, and BRIN indexes. B-tree is the default and handles equality and range queries."},
    {"id": 2,  "title": "Redis Caching Strategy", "body": "Redis supports strings, hashes, lists, sets, and sorted sets. Use TTL for automatic expiry. LRU eviction removes least recently used keys."},
    {"id": 3,  "title": "Kubernetes Pod Scheduling", "body": "The kube-scheduler assigns pods to nodes based on resource requests, affinities, taints, and tolerations."},
    {"id": 4,  "title": "Docker Networking", "body": "Docker provides bridge, host, and overlay network drivers. Bridge network is default for containers on the same host."},
    {"id": 5,  "title": "SQL Query Performance", "body": "EXPLAIN ANALYZE shows the query plan. Seq scan is slow on large tables. Index scan uses btree indexes for WHERE clause columns."},
    {"id": 6,  "title": "Python asyncio Basics", "body": "asyncio provides an event loop for concurrent I/O. Use async def for coroutines, await to pause execution, and asyncio.gather for parallel tasks."},
    {"id": 7,  "title": "pgvector Extension", "body": "pgvector adds vector similarity search to PostgreSQL. CREATE EXTENSION vector; then use vector type for storing embeddings."},
    {"id": 8,  "title": "Load Balancing Algorithms", "body": "Round-robin distributes evenly. Least-connections routes to the server with fewest active connections. IP hash ensures session persistence."},
    {"id": 9,  "title": "FastAPI Request Handling", "body": "FastAPI uses Pydantic models for request validation. Path parameters, query parameters, and request body all parsed automatically."},
    {"id": 10, "title": "Prometheus Metrics", "body": "Prometheus scrapes /metrics endpoints. Counter, Gauge, Histogram, and Summary are the four metric types. Labels add dimensions to metrics."},
    # BM25 should shine here — exact code identifiers
    {"id": 11, "title": "Error: CustomerNotFoundException", "body": "CustomerNotFoundException is raised when customer_id is not found in the database. Catch it in CustomerService.findById() and return 404."},
    {"id": 12, "title": "Config: QDRANT_API_KEY", "body": "Set QDRANT_API_KEY environment variable before starting the server. The key is validated on each request to the /collections endpoint."},
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def get_text(doc: dict) -> str:
    return f"{doc['title']} {doc['body']}"


# ─── Manual BM25 implementation (educational) ────────────────────────────────

class SimpleBM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize(d) for d in docs]
        self.n = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / self.n

        # Document frequency: how many docs contain each term
        self.df: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.n - n + 0.5) / (n + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        tf_map = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf(term) * (numerator / denominator if denominator else 0)
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        scores = [(i, self.score(tokens, doc)) for i, doc in enumerate(self.corpus)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== BM25 FULL-TEXT SEARCH DEMO ===\n")

    texts = [get_text(d) for d in DOCS]

    # Manual BM25 (no library)
    bm25_manual = SimpleBM25(texts)

    queries_semantic = [
        "how to speed up database queries",
        "running multiple tasks at the same time",
        "distributing traffic across multiple servers",
    ]

    queries_exact = [
        "CustomerNotFoundException",          # exact class name — BM25 wins
        "QDRANT_API_KEY",                     # exact env var — BM25 wins
        "CREATE EXTENSION vector",            # exact SQL — BM25 wins
        "EXPLAIN ANALYZE",                    # exact command
    ]

    print("─── Semantic queries (BM25 still works, vector search shines) ───\n")
    for query in queries_semantic:
        results = bm25_manual.search(query, top_k=3)
        print(f"  Q: {query!r}")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"    {rank}. score={score:.3f}  {DOCS[idx]['title']}")
        print()

    print("─── Exact / identifier queries (BM25 wins vs vector search) ───\n")
    for query in queries_exact:
        results = bm25_manual.search(query, top_k=3)
        print(f"  Q: {query!r}")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"    {rank}. score={score:.3f}  {DOCS[idx]['title']}")
        print()

    if BM25_AVAILABLE:
        print("─── rank-bm25 library comparison ───\n")
        tokenized_corpus = [tokenize(t) for t in texts]
        bm25_lib = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

        for query in queries_exact[:2]:
            tokens = tokenize(query)
            scores = bm25_lib.get_scores(tokens)
            top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Q: {query!r}")
            for rank, (idx, score) in enumerate(top, 1):
                print(f"    {rank}. score={score:.3f}  {DOCS[idx]['title']}")
            print()

    print("Key insight: BM25 is deterministic, fast, and requires no embeddings.")
    print("Use it as the first retrieval stage, then re-rank with vector similarity.")
