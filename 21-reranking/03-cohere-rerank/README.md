# Lab 03 — Cohere Rerank API

Cloud-based cross-encoder reranker. Accurate reranking without managing local GPU inference infrastructure.

## What you learn

- Two-stage retrieval pattern with cloud rerankers
- How cross-encoders evaluate full query-document attention
- Cost optimization: reducing context tokens fed into final LLM generators
- Fallback strategies for high-availability production RAG

## Run

```bash
python cohere_rerank.py
# Set COHERE_API_KEY to test live API; defaults to deterministic local fallback
```
