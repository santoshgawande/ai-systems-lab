# Lab 02 — Hybrid Search with RRF

Merge BM25 and vector rankings using Reciprocal Rank Fusion (RRF) for better RAG retrieval.

## What you learn

- Reciprocal Rank Fusion (RRF): the math and why it works
- How to combine BM25 and pgvector results into a single ranked list
- A/B eval: hybrid vs pure vector vs pure BM25 on a mixed query set
- When hybrid doesn't help (pure semantic questions)

## Run

```bash
python hybrid.py
# Uses rank-bm25 + Ollama for embeddings
```

## Reciprocal Rank Fusion (RRF)

```python
def rrf(results_a, results_b, k=60):
    scores = {}
    for rank, doc_id in enumerate(results_a):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(results_b):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

k=60 is empirically optimal (from the original RRF paper).
A document ranked #1 in both lists gets: 1/(60+1) + 1/(60+1) ≈ 0.033.
A document ranked #100 in both: 1/(60+101) × 2 ≈ 0.012.

## Why RRF beats score normalization

Score normalization (cosine + BM25 weighted sum) requires tuning the weight.
RRF only uses rank position — rank 1 always beats rank 2, regardless of score magnitude.
Works cross-system without hyperparameter tuning.

## Query types and which system wins

| Query | BM25 | Vector | Hybrid |
|---|---|---|---|
| "CustomerNotFoundException" | 1st | 3rd+ | 1st |
| "how to handle errors" | 3rd | 1st | 1st |
| "EXPLAIN ANALYZE slow query" | 1st | 2nd | 1st |
| "performance optimization" | 2nd | 1st | 1st |
