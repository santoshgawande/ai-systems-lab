"""
Section 21 — Reranking / Lab 03 — Cohere Rerank API

Cloud-based cross-encoder reranking:
  - Takes a query + list of document candidate strings
  - Evaluates joint cross-attention in the cloud
  - Returns ranked list with calibrated relevance scores (0.0 to 1.0)
  - Zero local GPU/VRAM requirement

Key advantages over local cross-encoders:
  1. No heavyweight PyTorch / CUDA runtime needed in microservices
  2. Sub-100ms latency on batches of 50–100 documents
  3. Multilingual support across 100+ languages (Cohere Rerank v3.5)

Run:
  COHERE_API_KEY="your-key" python cohere_rerank.py
  (Falls back gracefully to deterministic simulated cross-encoder if key is not set)
"""

import os
import math
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RerankResult:
    index: int
    document: str
    relevance_score: float


class CohereReranker:
    """Cohere Rerank v3 client with deterministic local cross-encoder fallback."""

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-v3.5"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[RerankResult]:
        if not documents:
            return []

        top_n = top_n or len(documents)

        # If API key is available, attempt real Cohere API request
        if self.api_key:
            try:
                import httpx
                resp = httpx.post(
                    "https://api.cohere.com/v1/rerank",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n
                    },
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for item in data.get("results", []):
                        idx = item["index"]
                        score = item["relevance_score"]
                        results.append(RerankResult(index=idx, document=documents[idx], relevance_score=score))
                    return results
            except Exception as e:
                print(f"[Warning] Cohere API call failed ({e}). Falling back to local scoring.")

        # Deterministic Cross-Encoder Simulation
        return self._simulate_cross_encoder_rerank(query, documents, top_n)

    def _simulate_cross_encoder_rerank(self, query: str, documents: List[str], top_n: int) -> List[RerankResult]:
        """Local heuristic simulating cross-encoder token interaction."""
        q_tokens = set(query.lower().split())
        scored = []

        for idx, doc in enumerate(documents):
            d_lower = doc.lower()
            d_tokens = d_lower.split()

            # 1. Exact match bonus
            exact_overlap = sum(1 for q in q_tokens if q in d_tokens)
            overlap_ratio = exact_overlap / max(1, len(q_tokens))

            # 2. Phrase continuity bonus
            phrase_bonus = 0.25 if query.lower() in d_lower else 0.0

            # 3. Simulated semantic depth
            char_match = sum(1 for c in query.lower() if c in d_lower) / max(1, len(query))
            raw_score = 0.5 * overlap_ratio + phrase_bonus + 0.25 * char_match
            calibrated_score = round(min(0.99, max(0.01, 1.0 / (1.0 + math.exp(-3.0 * (raw_score - 0.5))))), 4)

            scored.append(RerankResult(index=idx, document=doc, relevance_score=calibrated_score))

        # Sort descending by relevance score
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        return scored[:top_n]


def main():
    print("=" * 70)
    print("Lab 03: Cohere Rerank API (Two-Stage Cloud Cross-Encoder)")
    print("=" * 70)

    query = "How do I implement rate limiting in a distributed system?"

    # First-stage candidates (e.g. top-5 returned by fast bi-encoder vector search)
    candidates = [
        "Python asyncio event loop handling high concurrency tasks",
        "Distributed rate limiting with Redis sliding window and token bucket algorithms",
        "PostgreSQL table partitioning and b-tree indexing for query performance",
        "API Gateway rate limiter using Leaky Bucket in Nginx and Envoy",
        "How to scale microservices on Kubernetes with Horizontal Pod Autoscaler"
    ]

    print(f"\nQuery: \"{query}\"")
    print(f"\nFirst-Stage Candidates (Bi-Encoder Top-5):")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c}")

    reranker = CohereReranker()
    ranked = reranker.rerank(query, candidates, top_n=3)

    print(f"\n--- Cohere Rerank Top-3 (Cross-Encoder Re-ordered) ---")
    for rank, item in enumerate(ranked, 1):
        print(f"  Rank {rank} (Score: {item.relevance_score:.4f}) -> [{item.index}] {item.document}")


if __name__ == "__main__":
    main()
