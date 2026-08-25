"""
Section 21 — Reranking / Lab 05 — ColBERT Token-Level Late Interaction (MaxSim)

ColBERT (Khattab & Zaharia, SIGIR 2020; Santhanam et al. 2022) bridges the gap between:
  - Bi-Encoders (Fast, pre-computable embeddings, but loses fine-grained token-level nuances)
  - Cross-Encoders (Accurate joint attention, but requires quadratic forward passes per document)

Key Architectural Invariants:
  1. Token-Level Multi-Vector Encodings:
     Instead of pooling a document into a single 768-d vector, ColBERT retains a matrix of normalized token vectors:
       E_q \in \mathbb{R}^{|Q| \times d}, \quad E_d \in \mathbb{R}^{|D| \times d}
  2. Late Interaction MaxSim Operator:
     Computes relevance as the sum of maximum cosine similarities across all query tokens:
       S(q, d) = \sum_{i \in Q} \max_{j \in D} (E_{q, i} \cdot E_{d, j}^T)
  3. Extreme Efficiency:
     Document token matrices are pre-computed offline. At query time, MaxSim executes as fast matrix multiplications in SRAM.

Run:
  python colbert_maxsim.py
"""

import math
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ColBERTDocument:
    doc_id: str
    text: str
    token_embeddings: List[List[float]]  # [num_tokens, dim]


@dataclass
class ColBERTRerankScore:
    doc_id: str
    text: str
    maxsim_score: float
    token_alignments: List[Tuple[str, str, float]]  # (query_token, best_doc_token, similarity)


class ColBERTMaxSimReranker:
    """ColBERT Token-Level Multi-Vector Reranker implementing the MaxSim operator."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec)) + 1e-9
        return [round(x / norm, 4) for x in vec]

    @staticmethod
    def _dot(v1: List[float], v2: List[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    def _embed_tokens(self, text: str) -> Tuple[List[str], List[List[float]]]:
        """Contextual token embeddings for text."""
        tokens = text.lower().split()
        embeddings = []
        for idx, tok in enumerate(tokens):
            seed = sum((idx_c + 1) * ord(c) for idx_c, c in enumerate(tok))
            vec = [math.sin(seed * 0.73 + i * 1.37) for i in range(self.dim)]
            embeddings.append(self._normalize(vec))
        return tokens, embeddings

    def score_pair(self, query: str, document: str, doc_id: str = "doc") -> ColBERTRerankScore:
        """
        Computes ColBERT MaxSim Score: S(q, d) = sum_{i in Q} max_{j in D} (E_{q, i} . E_{d, j}^T)
        """
        q_tokens, q_embs = self._embed_tokens(query)
        d_tokens, d_embs = self._embed_tokens(document)

        total_maxsim = 0.0
        alignments = []

        for q_tok, q_vec in zip(q_tokens, q_embs):
            best_sim = -float('inf')
            best_doc_tok = ""

            for d_tok, d_vec in zip(d_tokens, d_embs):
                sim = self._dot(q_vec, d_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_doc_tok = d_tok

            total_maxsim += best_sim
            alignments.append((q_tok, best_doc_tok, round(best_sim, 4)))

        return ColBERTRerankScore(
            doc_id=doc_id,
            text=document,
            maxsim_score=round(total_maxsim, 4),
            token_alignments=alignments
        )

    def rerank(self, query: str, documents: List[Tuple[str, str]], top_n: int = 3) -> List[ColBERTRerankScore]:
        scores = [self.score_pair(query, doc_text, doc_id) for doc_id, doc_text in documents]
        scores.sort(key=lambda s: s.maxsim_score, reverse=True)
        return scores[:top_n]


def main():
    print("=" * 75)
    print("Lab 05: ColBERT Token-Level Late Interaction (MaxSim Reranker)")
    print("=" * 75)

    query = "redis token bucket rate limiter"
    documents = [
        ("doc_1", "Redis distributed cache and database replication strategies"),
        ("doc_2", "Token bucket rate limiter implementation using Redis Lua scripts"),
        ("doc_3", "Apache Kafka high throughput message broker partitions"),
        ("doc_4", "Leaky bucket and fixed window counters for API traffic control")
    ]

    print(f"Query: \"{query}\"\n")
    print("Candidate Documents:")
    for d_id, text in documents:
        print(f"  [{d_id}] {text}")

    reranker = ColBERTMaxSimReranker(dim=8)
    ranked = reranker.rerank(query, documents, top_n=3)

    print("\n--- ColBERT MaxSim Ranked Results ---")
    for rank, res in enumerate(ranked, 1):
        print(f"  Rank {rank} | Score: {res.maxsim_score} | [{res.doc_id}] {res.text}")
        print(f"    Token Alignments (MaxSim per Query Token):")
        for q_t, d_t, sim in res.token_alignments:
            print(f"      • '{q_t}' matched '{d_t}' (sim: {sim})")
        print()


if __name__ == "__main__":
    main()
