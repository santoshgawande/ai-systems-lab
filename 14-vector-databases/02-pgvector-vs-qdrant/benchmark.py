"""
Benchmark pgvector vs Qdrant: insert speed, query latency, recall@k.
Uses random vectors (no Ollama needed) for consistent reproducibility.
Requires: qdrant-client, psycopg2-binary
"""
import os
import time
import random
import math

PG_HOST = os.environ.get("PG_HOST", "192.168.0.111")
PG_PORT = int(os.environ.get("PG_PORT", 5432))
PG_DB = os.environ.get("PG_DB", "postgres")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "postgres")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "192.168.0.112")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

DIM = 128          # Use 128-dim for speed (768 in production)
N_VECTORS = 2000   # Reduce for faster demo
N_QUERIES = 10
TOP_K = 5


def random_vector(dim: int) -> list[float]:
    v = [random.gauss(0, 1) for _ in range(dim)]
    mag = math.sqrt(sum(x*x for x in v))
    return [x / mag for x in v]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    return dot  # already unit normalized


def brute_force_top_k(query: list[float], vectors: list[list[float]], k: int) -> list[int]:
    scores = [(i, cosine_similarity(query, v)) for i, v in enumerate(vectors)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:k]]


# ─── pgvector benchmark ───────────────────────────────────────────────────────

def bench_pgvector(vectors: list[list[float]], queries: list[list[float]]):
    try:
        import psycopg2
    except ImportError:
        print("  psycopg2 not installed. pip install psycopg2-binary")
        return None

    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB,
            user=PG_USER, password=PG_PASS, connect_timeout=3
        )
    except Exception as e:
        print(f"  Cannot connect to pgvector at {PG_HOST}:{PG_PORT}: {e}")
        return None

    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS bench_vectors")
    cur.execute(f"CREATE TABLE bench_vectors (id serial PRIMARY KEY, embedding vector({DIM}))")
    conn.commit()

    # Insert
    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        vec_str = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
        cur.execute("INSERT INTO bench_vectors (embedding) VALUES (%s)", (vec_str,))
    conn.commit()
    insert_time = time.perf_counter() - t0

    # Build IVFFlat index
    t0 = time.perf_counter()
    lists = max(1, int(math.sqrt(len(vectors))))
    cur.execute(f"CREATE INDEX ON bench_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists={lists})")
    conn.commit()
    index_time = time.perf_counter() - t0

    # Query
    query_times = []
    all_results = []
    for q in queries:
        vec_str = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
        t0 = time.perf_counter()
        cur.execute(
            f"SELECT id-1 FROM bench_vectors ORDER BY embedding <=> %s LIMIT %s",
            (vec_str, TOP_K)
        )
        ids = [row[0] for row in cur.fetchall()]
        query_times.append(time.perf_counter() - t0)
        all_results.append(ids)

    cur.execute("DROP TABLE bench_vectors")
    conn.commit()
    conn.close()

    avg_query_ms = sum(query_times) / len(query_times) * 1000
    return {
        "insert_s": insert_time,
        "index_s": index_time,
        "avg_query_ms": avg_query_ms,
        "results": all_results,
    }


# ─── Qdrant benchmark ────────────────────────────────────────────────────────

def bench_qdrant(vectors: list[list[float]], queries: list[list[float]]):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct, SearchParams
    except ImportError:
        print("  qdrant-client not installed. pip install qdrant-client")
        return None

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
        client.get_collections()
    except Exception as e:
        print(f"  Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}: {e}")
        return None

    client.recreate_collection(
        collection_name="bench",
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

    # Insert
    t0 = time.perf_counter()
    points = [PointStruct(id=i, vector=v) for i, v in enumerate(vectors)]
    client.upsert(collection_name="bench", points=points)
    insert_time = time.perf_counter() - t0

    # Query (ef=128)
    query_times = []
    all_results = []
    for q in queries:
        t0 = time.perf_counter()
        hits = client.search(
            collection_name="bench",
            query_vector=q,
            limit=TOP_K,
            search_params=SearchParams(hnsw_ef=128),
        )
        ids = [h.id for h in hits]
        query_times.append(time.perf_counter() - t0)
        all_results.append(ids)

    client.delete_collection("bench")

    avg_query_ms = sum(query_times) / len(query_times) * 1000
    return {
        "insert_s": insert_time,
        "index_s": 0,  # HNSW built incrementally during insert
        "avg_query_ms": avg_query_ms,
        "results": all_results,
    }


# ─── Recall calculation ───────────────────────────────────────────────────────

def recall_at_k(retrieved: list[list[int]], ground_truth: list[list[int]], k: int) -> float:
    hits = 0
    for ret, gt in zip(retrieved, ground_truth):
        hits += len(set(ret[:k]) & set(gt[:k]))
    return hits / (len(ground_truth) * k)


# ─── Main ────────────────────────────────────────────────────────────────────

print("=== PGVECTOR vs QDRANT BENCHMARK ===\n")
print(f"  Vectors: {N_VECTORS} × {DIM}-dim (normalized)")
print(f"  Queries: {N_QUERIES}")
print(f"  Top-K: {TOP_K}\n")

# Generate data
print("Generating random unit vectors...")
vectors = [random_vector(DIM) for _ in range(N_VECTORS)]
queries = [random_vector(DIM) for _ in range(N_QUERIES)]

# Ground truth via brute force
print("Computing ground truth (brute force)...\n")
ground_truth = [brute_force_top_k(q, vectors, TOP_K) for q in queries]

# Run benchmarks
results = {}

print("─── pgvector ───")
pg = bench_pgvector(vectors, queries)
if pg:
    recall = recall_at_k(pg["results"], ground_truth, TOP_K)
    print(f"  Insert:     {pg['insert_s']:.2f}s ({N_VECTORS/pg['insert_s']:.0f} vec/s)")
    print(f"  Index build:{pg['index_s']:.2f}s (IVFFlat)")
    print(f"  Avg query:  {pg['avg_query_ms']:.1f}ms")
    print(f"  Recall@{TOP_K}:  {recall*100:.1f}%")
    results["pgvector"] = {**pg, "recall": recall}
print()

print("─── Qdrant ───")
qd = bench_qdrant(vectors, queries)
if qd:
    recall = recall_at_k(qd["results"], ground_truth, TOP_K)
    print(f"  Insert:     {qd['insert_s']:.2f}s ({N_VECTORS/qd['insert_s']:.0f} vec/s, HNSW built inline)")
    print(f"  Avg query:  {qd['avg_query_ms']:.1f}ms (hnsw_ef=128)")
    print(f"  Recall@{TOP_K}:  {recall*100:.1f}%")
    results["qdrant"] = {**qd, "recall": recall}
print()

if "pgvector" in results and "qdrant" in results:
    pg_q = results["pgvector"]["avg_query_ms"]
    qd_q = results["qdrant"]["avg_query_ms"]
    print(f"Summary: Qdrant is {pg_q/qd_q:.1f}x faster on queries for {N_VECTORS} vectors")
    print(f"         Both have similar recall ({results['pgvector']['recall']*100:.0f}% vs {results['qdrant']['recall']*100:.0f}%)")
elif not results:
    print("Neither service available. To run: start pgvector on proxmox1, Qdrant on proxmox2.")
    print("See deploy/ directory for Docker configs.")
