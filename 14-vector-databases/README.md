# Section 14 — Vector Databases

Purpose-built stores for high-dimensional embeddings at production scale.

## What you learn

- Qdrant on your homelab (proxmox2:6333) — collections, upsert, search, delete
- pgvector vs Qdrant — when each wins
- Metadata filtering — combine structured filters with vector search
- Indexing strategies — HNSW, IVFFlat, flat — trade-offs

## Labs

| Lab | What it covers |
|---|---|
| 01-qdrant | Qdrant Python client: create collection, upsert vectors, search |
| 02-pgvector-vs-qdrant | Benchmark: insert/search speed, features comparison |
| 03-metadata-filtering | Filter by payload (category, date, score) + vector search |

## Setup

```bash
pip install -r requirements.txt
# Qdrant: docker running on proxmox2 (192.168.0.112:6333)
# PostgreSQL + pgvector: proxmox1 (192.168.0.111:5432)
```

## Qdrant vs pgvector

| | Qdrant | pgvector |
|---|---|---|
| Type | Dedicated VDB | PostgreSQL extension |
| Performance | Faster at pure vector search | Slower at scale |
| Filtering | Native payload + HNSW index | SQL WHERE + vector index |
| Persistence | Independent service | Shares PostgreSQL |
| Joins | No | Yes (any SQL) |
| Best for | High-QPS similarity search | RAG when you already have Postgres |

## Homelab

Qdrant is running on proxmox2: `http://192.168.0.112:6333`
Dashboard: `http://192.168.0.112:6333/dashboard`
