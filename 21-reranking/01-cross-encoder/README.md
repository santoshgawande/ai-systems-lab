# Lab 01 — Cross-Encoder Reranking

Use a cross-encoder model to rerank bi-encoder retrieval results for much higher precision.

## What you learn

- Why bi-encoder similarity search misses relevant results
- Cross-encoder: reads query + document together for true relevance scoring
- `sentence-transformers` cross-encoder models
- Two-stage pipeline: fast retrieval → accurate reranking

## Run

```bash
pip install sentence-transformers httpx
python rerank.py
```

## How cross-encoders work

**Bi-encoder** (standard vector search):
```
embed(query)  →  [0.2, 0.8, ...]   ← independent embeddings
embed(doc)    →  [0.3, 0.7, ...]
cosine_sim(query_vec, doc_vec)  ← fast, but no cross-attention
```

**Cross-encoder** (reranker):
```
[CLS] query [SEP] document [SEP]  ← concatenated input
→ transformer forward pass
→ single relevance score 0-1     ← reads them together = accurate
```

## Best models (free, run locally)

```
cross-encoder/ms-marco-MiniLM-L-6-v2   # 80MB, fast, good quality
cross-encoder/ms-marco-MiniLM-L-12-v2  # 130MB, slower, better
BAAI/bge-reranker-base                  # Strong, multilingual
```

## Cohere Rerank API (cloud alternative)

```python
import cohere
co = cohere.Client(os.environ["COHERE_API_KEY"])

results = co.rerank(
    model="rerank-english-v3.0",
    query="how to handle database errors",
    documents=[doc["text"] for doc in candidates],
    top_n=5,
)
for r in results.results:
    print(r.relevance_score, candidates[r.index]["title"])
```
