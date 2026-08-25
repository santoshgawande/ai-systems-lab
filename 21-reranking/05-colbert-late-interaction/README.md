# Lab 05 — ColBERT Token-Level Late Interaction (MaxSim)

Token-level multi-vector late interaction reranking. Combines the accuracy of cross-encoders with the efficiency of pre-computed embeddings.

## What you learn

- The trade-offs of Bi-Encoders, Cross-Encoders, and Late-Interaction architectures
- The MaxSim operator: $S(q, d) = \sum_{i \in Q} \max_{j \in D} (E_{q, i} \cdot E_{d, j}^T)$
- Fine-grained token alignment without full quadratic cross-attention
- How ColBERTv2 and RAGatouille power high-throughput search

## Run

```bash
python colbert_maxsim.py
```
