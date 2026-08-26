"""
Qdrant metadata filtering: combine vector search with structured payload filters.
Shows filter DSL: match, range, must/should/must_not.
Requires: qdrant-client, Ollama at localhost:11434
"""
import os
import httpx

QDRANT_HOST = os.environ.get("QDRANT_HOST", "192.168.0.112")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "lab-docs"


def embed(text: str) -> list[float]:
    r = httpx.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    return r.json()["embedding"]


DOCS = [
    {"id": 1, "title": "PostgreSQL Indexes",        "category": "database", "product": "backend",  "difficulty": "intermediate", "published": True,  "views": 3200, "tags": ["sql", "performance"]},
    {"id": 2, "title": "Redis Pub/Sub",              "category": "cache",    "product": "backend",  "difficulty": "advanced",     "published": True,  "views": 890,  "tags": ["redis", "messaging"]},
    {"id": 3, "title": "K8s Rolling Updates",        "category": "devops",   "product": "platform", "difficulty": "advanced",     "published": True,  "views": 4100, "tags": ["kubernetes", "deployment"]},
    {"id": 4, "title": "Docker Compose Basics",      "category": "devops",   "product": "platform", "difficulty": "beginner",     "published": True,  "views": 5600, "tags": ["docker", "containers"]},
    {"id": 5, "title": "pgvector Setup Guide",       "category": "database", "product": "backend",  "difficulty": "beginner",     "published": True,  "views": 1200, "tags": ["vectors", "postgresql"]},
    {"id": 6, "title": "Qdrant Production Deploy",   "category": "database", "product": "platform", "difficulty": "advanced",     "published": True,  "views": 720,  "tags": ["vectors", "qdrant"]},
    {"id": 7, "title": "API Rate Limiting Patterns", "category": "backend",  "product": "backend",  "difficulty": "intermediate", "published": True,  "views": 2800, "tags": ["api", "performance"]},
    {"id": 8, "title": "Monitoring with Prometheus", "category": "devops",   "product": "platform", "difficulty": "intermediate", "published": True,  "views": 1900, "tags": ["monitoring", "metrics"]},
    {"id": 9, "title": "SQL Query Draft",            "category": "database", "product": "backend",  "difficulty": "beginner",     "published": False, "views": 0,    "tags": ["sql", "draft"]},
    {"id": 10,"title": "Nginx Load Balancer Config", "category": "devops",   "product": "platform", "difficulty": "intermediate", "published": True,  "views": 3400, "tags": ["nginx", "loadbalancer"]},
]

BODY_MAP = {
    1:  "Database indexes dramatically speed up SELECT queries by allowing the engine to skip full table scans.",
    2:  "Redis pub/sub enables event-driven architectures with channels, subscribers, and message fanout.",
    3:  "Kubernetes rolling updates replace pods gradually to ensure zero-downtime deployments.",
    4:  "Docker Compose defines multi-container applications with a simple YAML file.",
    5:  "pgvector extends PostgreSQL with vector storage and similarity search using cosine or L2 distance.",
    6:  "Qdrant is a dedicated vector database with HNSW indexing and a full REST API.",
    7:  "Rate limiting prevents API abuse through token bucket, sliding window, and fixed window algorithms.",
    8:  "Prometheus scrapes metrics endpoints and stores time-series data for alerting and dashboards.",
    9:  "DRAFT: comparing join strategies in PostgreSQL — hash join vs nested loop vs merge join.",
    10: "Nginx can distribute traffic across multiple backend servers using round-robin or least-connections.",
}


def show_results(results, label: str):
    print(f"\n  {label}:")
    if not results:
        print("    (no results)")
    for r in results:
        p = r.payload
        status = "" if p["published"] else " [DRAFT]"
        print(f"    {r.score:.3f}  [{p['category']}/{p['difficulty']}]  views={p['views']}  {p['title']}{status}")


if __name__ == "__main__":
    print("=== METADATA FILTERING DEMO ===\n")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct,
            Filter, FieldCondition, MatchValue, MatchAny, Range
        )
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
        client.get_collections()
        connected = True
    except Exception as e:
        connected = False
        print(f"Qdrant not available: {e}")

    if not connected:
        print("Showing filter syntax:\n")
        print("""
# Exact match
Filter(must=[FieldCondition(key="category", match=MatchValue(value="database"))])

# Match any of multiple values (IN)
Filter(must=[FieldCondition(key="category", match=MatchAny(any=["database", "cache"]))])

# Numeric range
Filter(must=[FieldCondition(key="views", range=Range(gte=1000))])

# Boolean
Filter(must=[FieldCondition(key="published", match=MatchValue(value=True))])

# AND (multiple must conditions)
Filter(must=[
    FieldCondition(key="product", match=MatchValue(value="backend")),
    FieldCondition(key="published", match=MatchValue(value=True)),
])

# NOT
Filter(must_not=[FieldCondition(key="published", match=MatchValue(value=False))])

# OR (should)
Filter(should=[
    FieldCondition(key="difficulty", match=MatchValue(value="beginner")),
    FieldCondition(key="difficulty", match=MatchValue(value="intermediate")),
])
""")
    else:
        dim = len(embed("test"))

        # Setup
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        print(f"Embedding {len(DOCS)} documents...")
        points = []
        for doc in DOCS:
            text = f"{doc['title']}. {BODY_MAP[doc['id']]}"
            points.append(PointStruct(
                id=doc["id"],
                vector=embed(text),
                payload={k: v for k, v in doc.items() if k != "id"}
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Upserted {len(points)} points.\n")

        query = "database performance and query optimization"
        qvec = embed(query)
        print(f"Query: {query!r}\n")

        # 1. No filter
        results = client.search(COLLECTION, query_vector=qvec, limit=5, with_payload=True)
        show_results(results, "No filter (top 5)")

        # 2. Category filter
        results = client.search(
            COLLECTION, query_vector=qvec, limit=5, with_payload=True,
            query_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="database"))])
        )
        show_results(results, "Filter: category=database")

        # 3. Published only (exclude drafts)
        results = client.search(
            COLLECTION, query_vector=qvec, limit=5, with_payload=True,
            query_filter=Filter(must_not=[FieldCondition(key="published", match=MatchValue(value=False))])
        )
        show_results(results, "Filter: published only (exclude drafts)")

        # 4. High-traffic intermediate articles
        results = client.search(
            COLLECTION, query_vector=qvec, limit=5, with_payload=True,
            query_filter=Filter(must=[
                FieldCondition(key="views", range=Range(gte=2000)),
                FieldCondition(key="difficulty", match=MatchValue(value="intermediate")),
            ])
        )
        show_results(results, "Filter: views>=2000 AND difficulty=intermediate")

        # 5. Backend or cache docs (OR)
        results = client.search(
            COLLECTION, query_vector=qvec, limit=5, with_payload=True,
            query_filter=Filter(should=[
                FieldCondition(key="category", match=MatchValue(value="database")),
                FieldCondition(key="category", match=MatchValue(value="cache")),
            ])
        )
        show_results(results, "Filter: category=database OR cache (should)")

        client.delete_collection(COLLECTION)
        print("\nFiltering lets you scope vector search to a subset without losing accuracy.")
