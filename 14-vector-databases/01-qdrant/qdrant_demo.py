"""
Qdrant vector database: create collection, upsert documents, search, filter.
Connects to Qdrant on proxmox2 (192.168.0.112:6333).
Embeddings via Ollama (localhost:11434).
"""
import os
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    HnswConfigDiff, SearchParams
)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "192.168.0.112")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "lab-articles"
VECTOR_DIM = 768


def embed(text: str) -> list[float]:
    r = httpx.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    return r.json()["embedding"]


def check_connection(client: QdrantClient) -> bool:
    try:
        client.get_collections()
        return True
    except Exception:
        return False


# ─── Sample data ─────────────────────────────────────────────────────────────

ARTICLES = [
    {
        "id": 1, "title": "Getting Started with PostgreSQL",
        "body": "PostgreSQL is a powerful open-source relational database. Install with brew install postgresql.",
        "category": "database", "difficulty": "beginner", "views": 1200,
    },
    {
        "id": 2, "title": "Redis Caching Patterns",
        "body": "Redis excels at caching, pub/sub, and session storage. Use TTL to expire stale data automatically.",
        "category": "cache", "difficulty": "intermediate", "views": 890,
    },
    {
        "id": 3, "title": "Kubernetes Deployment Strategies",
        "body": "Blue-green, canary, and rolling updates are the three main K8s deployment strategies.",
        "category": "devops", "difficulty": "advanced", "views": 2300,
    },
    {
        "id": 4, "title": "Docker Networking Explained",
        "body": "Bridge, host, and overlay networks serve different purposes in Docker environments.",
        "category": "devops", "difficulty": "intermediate", "views": 1500,
    },
    {
        "id": 5, "title": "SQL Query Optimization",
        "body": "Use EXPLAIN ANALYZE to profile queries. Index columns in WHERE and JOIN clauses.",
        "category": "database", "difficulty": "intermediate", "views": 3100,
    },
    {
        "id": 6, "title": "Introduction to pgvector",
        "body": "pgvector adds vector similarity search to PostgreSQL. Store embeddings alongside relational data.",
        "category": "database", "difficulty": "intermediate", "views": 670,
    },
    {
        "id": 7, "title": "Qdrant vs Pinecone vs Weaviate",
        "body": "Purpose-built vector databases offer better performance than pgvector at large scale.",
        "category": "database", "difficulty": "advanced", "views": 4200,
    },
    {
        "id": 8, "title": "CI/CD Pipeline Design",
        "body": "A good CI/CD pipeline runs tests, builds images, and deploys to staging automatically.",
        "category": "devops", "difficulty": "intermediate", "views": 1800,
    },
]


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== QDRANT DEMO ===\n")

try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
    connected = check_connection(client)
except Exception:
    connected = False

if not connected:
    print(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}")
    print("Showing Qdrant API shapes:\n")
    print("""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="192.168.0.112", port=6333)

# Create collection
client.recreate_collection(
    collection_name="articles",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Upsert points (vectors + payload)
client.upsert(
    collection_name="articles",
    points=[
        PointStruct(
            id=1,
            vector=embed("PostgreSQL is a relational database"),
            payload={"title": "PostgreSQL Guide", "category": "database"}
        )
    ]
)

# Search
results = client.search(
    collection_name="articles",
    query_vector=embed("how to speed up database queries"),
    limit=5,
    with_payload=True,
)
for r in results:
    print(f"Score: {r.score:.3f} | {r.payload['title']}")

# Search with filter
from qdrant_client.models import Filter, FieldCondition, MatchValue
results = client.search(
    collection_name="articles",
    query_vector=embed("database optimization"),
    query_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="database"))]),
    limit=3,
)
""")
else:
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}\n")

    # Create (or recreate) collection
    print(f"Creating collection '{COLLECTION}'...")
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )
    print("  Done.\n")

    # Embed and upsert all articles
    print(f"Embedding and upserting {len(ARTICLES)} articles...")
    points = []
    for article in ARTICLES:
        text = f"{article['title']}. {article['body']}"
        vector = embed(text)
        points.append(PointStruct(
            id=article["id"],
            vector=vector,
            payload={
                "title": article["title"],
                "category": article["category"],
                "difficulty": article["difficulty"],
                "views": article["views"],
            }
        ))
        print(f"  {article['id']}. {article['title']}")

    client.upsert(collection_name=COLLECTION, points=points)
    info = client.get_collection(COLLECTION)
    print(f"\nCollection stats: {info.points_count} points, {info.vectors_count} vectors\n")

    # Basic similarity search
    queries = [
        "how to optimize database queries",
        "container orchestration and deployment",
        "storing and searching vectors",
    ]

    print("─── Similarity Search ───\n")
    for query in queries:
        qvec = embed(query)
        results = client.search(
            collection_name=COLLECTION,
            query_vector=qvec,
            limit=3,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )
        print(f"Query: {query!r}")
        for r in results:
            print(f"  {r.score:.3f}  [{r.payload['category']}/{r.payload['difficulty']}]  {r.payload['title']}")
        print()

    # Filtered search
    print("─── Filtered Search (category=database) ───\n")
    qvec = embed("performance and speed")
    results = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="database"))]
        ),
        with_payload=True,
    )
    for r in results:
        print(f"  {r.score:.3f}  views={r.payload['views']}  {r.payload['title']}")
    print()

    # Range filter: popular articles (views > 2000)
    print("─── Range Filter (views > 2000) ───\n")
    results = client.search(
        collection_name=COLLECTION,
        query_vector=embed("infrastructure and deployment"),
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="views", range=Range(gt=2000))]
        ),
        with_payload=True,
    )
    for r in results:
        print(f"  {r.score:.3f}  views={r.payload['views']}  {r.payload['title']}")

    print(f"\nDashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")
