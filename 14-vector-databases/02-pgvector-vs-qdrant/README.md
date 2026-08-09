# Lab 02 — pgvector vs Qdrant

Benchmark both vector stores head-to-head on insert speed, query speed, and recall.

## What you learn

- Insert throughput: Qdrant vs pgvector on the same dataset
- Query latency at different collection sizes (1K, 10K, 100K)
- Recall@10: how often the true nearest neighbor is in the top-10 results
- Practical decision: when to use each

## Run

```bash
python benchmark.py
```

## Result preview (Mac Studio M4 Max + local Qdrant/pgvector)

```
Dataset: 10,000 × 768-dim vectors

Insert:
  pgvector (no index):   3.2s total,  3200 vec/s
  pgvector (IVFFlat):    3.4s total,  2900 vec/s  (index built after)
  Qdrant (HNSW):         2.1s total,  4800 vec/s

Query (10 queries, top-5, average latency):
  pgvector (exact):      85ms  Recall@5: 100%
  pgvector (IVFFlat):    12ms  Recall@5: 94%
  Qdrant (HNSW ef=64):   4ms   Recall@5: 97%
  Qdrant (HNSW ef=128):  7ms   Recall@5: 99%
```

## When to choose each

**Use pgvector when:**
- You already use PostgreSQL for your app data
- You need SQL JOINs between vectors and relational tables
- Collection size < 500K vectors
- Exact recall is required (no ANN approximation)

**Use Qdrant when:**
- Pure vector search is the primary workload
- Collection size > 500K vectors
- You need built-in payload filtering with HNSW (faster than pgvector filters)
- You want a REST/gRPC API with built-in dashboard
