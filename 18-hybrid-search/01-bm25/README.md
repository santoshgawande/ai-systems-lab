# Lab 01 — BM25 Full-Text Search

BM25 (Best Match 25) is the gold standard full-text ranking algorithm — used by Elasticsearch, Solr, and most search engines.

## What you learn

- How BM25 scores documents: TF saturation + IDF + field length normalization
- Why BM25 beats TF-IDF on longer documents
- Using `rank-bm25` library for Python
- Where BM25 wins against vector search (exact keywords, rare terms, code)

## Run

```bash
pip install rank-bm25
python bm25.py
```

## BM25 formula

```
BM25(q, d) = Σ IDF(qi) × (tf(qi,d) × (k1+1)) / (tf(qi,d) + k1×(1-b+b×|d|/avgdl))

Where:
  tf(qi,d)  = term frequency of qi in document d
  IDF(qi)   = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
  |d|       = document length
  avgdl     = average document length in corpus
  k1 = 1.5  # term frequency saturation (higher = more weight to TF)
  b  = 0.75 # length normalization (1.0 = full normalization, 0 = none)
```

## BM25 vs TF-IDF vs Vector search

| Method | Exact match | Semantic | Rare terms | Multilingual | Speed |
|---|---|---|---|---|---|
| BM25 | ✓✓ | ✗ | ✓✓ | ✓ | Very fast |
| TF-IDF | ✓ | ✗ | ✓ | ✓ | Very fast |
| Vector | ✗ | ✓✓ | ✗ | ✓✓ (if multilingual model) | Fast |
| Hybrid | ✓✓ | ✓✓ | ✓✓ | ✓✓ | Fast |
