# Lab 02 — MMR (Maximal Marginal Relevance)

Retrieve diverse results instead of near-duplicates. Solves the "5 similar answers" problem.

## What you learn

- Why cosine-only retrieval returns redundant results
- MMR formula: `lambda * sim(query, doc) - (1-lambda) * max(sim(selected, doc))`
- lambda trade-off: 1.0 = pure relevance, 0.0 = pure diversity
- When to use MMR in production RAG

## Run

```bash
pip install httpx
python mmr.py
# Requires Ollama at localhost:11434 with nomic-embed-text
```

## The MMR algorithm

```python
selected = []
remaining = all_documents

for i in range(top_k):
    best = argmax over remaining:
        lambda * sim(query, doc) - (1 - lambda) * max(sim(s, doc) for s in selected)
    selected.append(best)
    remaining.remove(best)
```

**Intuition**: each iteration picks the doc that's most relevant AND least like what you already picked.

## Lambda guide

| lambda | Behaviour | Use case |
|--------|-----------|----------|
| 1.0 | Pure relevance | When docs are already diverse |
| 0.7 | Mostly relevant, some diversity | General RAG (recommended) |
| 0.5 | Balanced | When corpus has many near-duplicates |
| 0.3 | Mostly diverse | Exploratory search |
| 0.0 | Pure diversity | Corpus sampling / data exploration |

## When MMR matters

**Without MMR** — top-3 for "Python error handling":
1. Use try/except to catch exceptions
2. Catch specific exceptions, not bare except
3. Python exception handling best practices

→ All three say the same thing.

**With MMR (lambda=0.5)** — top-3:
1. Use try/except to catch exceptions ← most relevant
2. Database connection error handling ← different cluster
3. Logging exceptions with structlog ← different angle

→ Better answer coverage for the LLM to synthesise from.
