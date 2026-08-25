"""
Section 21 — Reranking / Lab 04 — Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion (RRF) combines rankings from multiple retrieval systems
(e.g., Dense Vector Search, BM25 Keyword Search, and SPLADE sparse representations)
without requiring calibrated score normalization.

Mathematical Formula:
  RRF_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}

where:
  - M is the set of retrieval models
  - r_m(d) is the 1-based rank position of document d in model m's output
  - k is a smoothing constant (default: k = 60, Cormack et al. 2009)

Key Advantages:
  1. No scale mismatch: Fuses BM25 (unbounded [0, \inf)) + Dense Cosine ([-1, 1]) + Cross-Encoder logits seamlessly.
  2. Zero hyperparameter tuning: k = 60 is robust across virtually all domains.
  3. Resistance to score outliers: An anomalous 99.9% similarity score on one retriever cannot overwhelm other systems.

Run:
  python rrf.py
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RankedItem:
    doc_id: str
    content: str
    rank: int


@dataclass
class RRFResult:
    doc_id: str
    content: str
    rrf_score: float
    system_ranks: Dict[str, int]


def reciprocal_rank_fusion(
    system_rankings: Dict[str, List[RankedItem]],
    k: int = 60,
    top_n: int = 5
) -> List[RRFResult]:
    """
    Combines multiple ranked lists into a single consensus ranking using RRF.
    """
    scores: Dict[str, float] = {}
    doc_contents: Dict[str, str] = {}
    doc_system_ranks: Dict[str, Dict[str, int]] = {}

    for sys_name, ranking in system_rankings.items():
        for item in ranking:
            doc_id = item.doc_id
            rank = item.rank
            doc_contents[doc_id] = item.content

            if doc_id not in doc_system_ranks:
                doc_system_ranks[doc_id] = {}
            doc_system_ranks[doc_id][sys_name] = rank

            # Compute RRF score contribution: 1 / (k + rank)
            contribution = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution

    # Sort descending by RRF score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in sorted_docs[:top_n]:
        results.append(RRFResult(
            doc_id=doc_id,
            content=doc_contents[doc_id],
            rrf_score=round(score, 6),
            system_ranks=doc_system_ranks[doc_id]
        ))

    return results


def main():
    print("=" * 75)
    print("Lab 04: Reciprocal Rank Fusion (RRF) Multi-Retriever Consensus")
    print("=" * 75)

    # Retrieval System 1: Dense Vector Bi-Encoder (Semantic search)
    dense_results = [
        RankedItem("doc_B", "Token Bucket and Leaky Bucket algorithms for API rate limiting", 1),
        RankedItem("doc_A", "Distributed rate limiting with Redis sliding window", 2),
        RankedItem("doc_C", "Nginx reverse proxy rate limit directives", 3),
        RankedItem("doc_D", "Database query optimization and connection pooling", 4),
    ]

    # Retrieval System 2: BM25 Sparse Keyword Search (Lexical match)
    bm25_results = [
        RankedItem("doc_A", "Distributed rate limiting with Redis sliding window", 1),
        RankedItem("doc_E", "Redis cluster failover and persistence configuration", 2),
        RankedItem("doc_B", "Token Bucket and Leaky Bucket algorithms for API rate limiting", 3),
        RankedItem("doc_C", "Nginx reverse proxy rate limit directives", 4),
    ]

    # Retrieval System 3: SPLADE / Learned Sparse Representation
    splade_results = [
        RankedItem("doc_A", "Distributed rate limiting with Redis sliding window", 1),
        RankedItem("doc_B", "Token Bucket and Leaky Bucket algorithms for API rate limiting", 2),
        RankedItem("doc_F", "Envoy proxy distributed rate limit gRPC service", 3),
        RankedItem("doc_C", "Nginx reverse proxy rate limit directives", 4),
    ]

    system_runs = {
        "Dense_Vector": dense_results,
        "BM25_Keyword": bm25_results,
        "SPLADE_Sparse": splade_results
    }

    fused = reciprocal_rank_fusion(system_runs, k=60, top_n=5)

    print("\n--- Fused Rankings via RRF (k = 60) ---")
    for rank, item in enumerate(fused, 1):
        ranks_str = ", ".join(f"{k}: #{v}" for k, v in item.system_ranks.items())
        print(f"  Rank {rank} (Score: {item.rrf_score:.6f}) [{item.doc_id}]")
        print(f"         Content : {item.content}")
        print(f"         Sources : {ranks_str}\n")


if __name__ == "__main__":
    main()
