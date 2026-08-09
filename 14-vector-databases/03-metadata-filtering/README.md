# Lab 03 — Metadata Filtering

Combine vector similarity with structured payload filters — the key to production RAG.

## What you learn

- Pre-filtering vs post-filtering — which is faster and why Qdrant does pre-filtering right
- Filtering by string, enum, range, and nested fields
- `must`, `should`, `must_not` — Qdrant's boolean filter DSL
- How filters affect recall (fewer candidates = potentially lower recall)

## Run

```bash
python filtering.py
```

## Why filtering matters

Without filtering, similarity search returns globally similar results.
With filtering, you scope to a subset first, then find the most similar.

Example: "How do I reset my password?"
- Without filter: returns any password-reset docs from any product
- With `filter(product=customer_portal)`: returns docs for the correct product

## Filter DSL

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

# Match exact value
Filter(must=[FieldCondition(key="category", match=MatchValue(value="database"))])

# Match any of several values (like SQL IN)
Filter(must=[FieldCondition(key="status", match=MatchAny(any=["published", "featured"]))])

# Numeric range
Filter(must=[FieldCondition(key="views", range=Range(gte=1000, lt=10000))])

# Combine with AND (must)
Filter(must=[
    FieldCondition(key="category", match=MatchValue(value="devops")),
    FieldCondition(key="difficulty", match=MatchValue(value="intermediate")),
])

# Combine with OR (should)
Filter(should=[
    FieldCondition(key="category", match=MatchValue(value="database")),
    FieldCondition(key="category", match=MatchValue(value="cache")),
])

# Exclude (must_not)
Filter(must_not=[FieldCondition(key="draft", match=MatchValue(value=True))])
```

## HNSW + filter performance

Qdrant pre-filters with HNSW: payload index is built alongside the vector index.
Filtering to 10% of vectors = nearly the same speed as filtering to 100%.

pgvector post-filters: runs vector search then applies WHERE clause.
Filtering in pgvector requires the query to scan more vectors before finding k valid results.
