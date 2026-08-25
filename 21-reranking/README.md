# Section 21 — Reranking

First-stage retrieval gives you the top-50. Reranking finds the actual top-5.

## What you learn

- Why similarity search alone gives poor precision at top-k
- Cross-encoder reranking: slower but far more accurate than bi-encoder
- MMR (Maximal Marginal Relevance): diverse results instead of near-duplicates
- Cohere Rerank API: cloud cross-encoder, no model to run locally

## Labs

| Lab | What it covers |
|---|---|
| 01-cross-encoder | Cross-encoder vs bi-encoder, sentence-transformers reranking |
| 02-mmr | MMR algorithm, diversity vs relevance trade-off |
| 03-cohere-rerank | Cloud cross-encoder API, batch reranking, fallback strategies |
| 04-reciprocal-rank-fusion | Reciprocal Rank Fusion (RRF), multi-retriever hybrid consensus |

## Setup

```bash
pip install -r requirements.txt
# Ollama at localhost:11434 for embeddings
```

## Two-stage retrieval

```
Stage 1: Fast bi-encoder (cosine similarity)
  Query → embed → top-50 candidates in <10ms
  (Fast but imprecise: "bank" matches "river bank" and "bank account")

Stage 2: Cross-encoder reranker
  Each of 50 candidates → joint encode with query → relevance score
  Returns true top-5 in ~200ms
  (Slow but precise: reads query+doc together, understands context)
```

## Cross-encoder vs bi-encoder

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Speed | Fast (embed once, dot product) | Slow (N forward passes) |
| Accuracy | Good | Excellent |
| Use in | Stage 1: candidate retrieval | Stage 2: reranking |
| Models | nomic-embed-text, text-embedding-3 | cross-encoder/ms-marco-MiniLM |
