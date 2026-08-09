# Lab 01 — Qdrant

Connect to your homelab Qdrant instance and run vector search.

## What you learn

- Creating a collection with HNSW index and cosine distance
- Upserting points (vectors + payload metadata)
- Searching by vector similarity
- Deleting and updating points
- Qdrant's dashboard UI

## Run

```bash
# Make sure Qdrant is running on proxmox2
python qdrant_demo.py
```

## Homelab connection

```
Qdrant: http://192.168.0.112:6333
Dashboard: http://192.168.0.112:6333/dashboard
Ollama (embeddings): http://localhost:11434
```

## Key Qdrant concepts

```
Collection  = a named index (like a table)
Point       = {id, vector, payload}  (like a row)
Payload     = arbitrary JSON metadata attached to each vector
Filter      = narrow search by payload fields before/after vector search
```

## HNSW index parameters

```python
HnswConfigDiff(
    m=16,               # number of bidirectional links (higher = better recall, more RAM)
    ef_construct=100,   # build-time accuracy (higher = slower build, better index)
)

# Search-time:
search_params=SearchParams(hnsw_ef=128)  # higher = better recall, slower query
```
