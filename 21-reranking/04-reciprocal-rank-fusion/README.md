# Lab 04 — Reciprocal Rank Fusion (RRF)

Combine rankings from multiple search systems (Dense, BM25, SPLADE) into a single consensus ranking.

## What you learn

- Why raw score interpolation ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) fails without normalization
- The RRF formula: $RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$
- Why $k = 60$ is the standard smoothing factor (Cormack et al.)
- How Elasticsearch, Pinecone, and Azure AI Search implement hybrid search with RRF

## Run

```bash
python rrf.py
```
