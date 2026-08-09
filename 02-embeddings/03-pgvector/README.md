# Lab 03 — pgvector: Embeddings in PostgreSQL

Store vectors in PostgreSQL, create an ANN index, and run similarity queries — exactly how RAG systems work in production.

## What you learn

- How to enable the `vector` extension in PostgreSQL
- How to store embeddings as a `vector(768)` column
- How to query with `<=>` (cosine distance operator)
- How IVFFlat index speeds up approximate nearest-neighbor search

## Requirements

PostgreSQL + pgvector running on proxmox1:

```bash
# proxmox1 docker-compose already includes pgvector — check deploy/proxmox1/
docker exec -it postgres psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Run

```bash
python store.py
```

## Key SQL

```sql
-- Store a vector
INSERT INTO documents (content, embedding) VALUES ('text', '[0.1, 0.2, ...]'::vector);

-- Similarity search (cosine distance)
SELECT content, 1 - (embedding <=> query_vec::vector) AS similarity
FROM documents
ORDER BY embedding <=> query_vec::vector
LIMIT 5;

-- ANN index for scale (create after bulk insert)
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```
