# Lab 02 — Comparing Embedding Models

Different models produce vectors of different dimensions and quality. Learn how to evaluate them.

## What you learn

- How embedding dimensions affect expressiveness vs speed
- Why a "separation gap" (similar - different) measures model quality
- Trade-offs: `nomic-embed-text` (768d) vs `mxbai-embed-large` (1024d) vs `all-minilm` (384d)

## Run

```bash
# Pull extra models first (optional — nomic alone works)
ollama pull mxbai-embed-large
ollama pull all-minilm

python compare.py
```

## What to look for

A good embedding model produces:
- HIGH similarity for semantically related sentences (~0.85+)
- LOW similarity for unrelated sentences (~0.3 or less)
- LARGE separation gap = better model
