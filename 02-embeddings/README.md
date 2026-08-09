# 02 — Embeddings

Dense vector representations of text. The foundation of semantic search, RAG, and memory systems.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires:
- Ollama at `http://localhost:11434` with `nomic-embed-text` pulled
- PostgreSQL + pgvector at `proxmox1:5432` (lab 03 only)

```bash
ollama pull nomic-embed-text
```

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-what-are-embeddings/` | Text → vector, cosine similarity, nearest neighbors | `python embed.py` |
| `02-embedding-models/` | Compare models: dimensions, speed, quality trade-offs | `python compare.py` |
| `03-pgvector/` | Store embeddings in PostgreSQL, run similarity queries | `python store.py` |

## Key concepts

- Embeddings map semantically similar text to nearby points in vector space
- Cosine similarity measures the angle between two vectors (1.0 = identical meaning, 0.0 = unrelated)
- `nomic-embed-text` produces 768-dimensional vectors — each sentence becomes a list of 768 floats
- Vector stores (pgvector, Qdrant, Chroma) use ANN indexes (IVFFlat, HNSW) for fast similarity search at scale
