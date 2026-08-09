# Section 18 — Hybrid Search

Combine full-text BM25 with vector similarity for better RAG retrieval than either alone.

## What you learn

- BM25 — the TF-IDF successor powering most search engines
- Why vector search alone misses exact keyword matches
- Reciprocal Rank Fusion (RRF) — merge BM25 + vector rankings
- When hybrid beats pure vector, and when it doesn't

## Labs

| Lab | What it covers |
|---|---|
| 01-bm25 | BM25 from scratch, rank-bm25, exact keyword recall |
| 02-hybrid-fusion | RRF combining BM25 + vector, A/B eval against pure vector |

## Setup

```bash
pip install -r requirements.txt
# Ollama at localhost:11434 for embeddings
# pgvector at proxmox1 for vector search
```

## Why hybrid matters

**Vector search** is great for semantic similarity:
- Query: "how to handle errors" → finds "exception handling guide" ✓

**But misses exact matches:**
- Query: "CustomerNotFoundException" → may not find docs mentioning that exact class ✗

**BM25** catches exact terms but misses semantics:
- Query: "how to handle errors" → might miss "exception handling" ✗
- Query: "CustomerNotFoundException" → finds it immediately ✓

**Hybrid = best of both.**

## Reciprocal Rank Fusion (RRF)

```python
def rrf(results_a: list, results_b: list, k: int = 60) -> list:
    scores = {}
    for rank, doc_id in enumerate(results_a):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(results_b):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

## When hybrid wins

| Query type | Pure vector | BM25 | Hybrid |
|---|---|---|---|
| Semantic concept | ✓ | ✗ | ✓ |
| Exact code/name | ✗ | ✓ | ✓ |
| Mixed intent | ✓ | ✓ | ✓✓ |
