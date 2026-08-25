import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-mmr"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "03-cohere-rerank"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "04-reciprocal-rank-fusion"))

from mmr import mmr, cosine, greedy_top_k
from cohere_rerank import CohereReranker
from rrf import reciprocal_rank_fusion, RankedItem


class TestReranking(unittest.TestCase):
    def test_cosine_similarity(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        self.assertAlmostEqual(cosine(v1, v2), 1.0)

    def test_mmr_diversification(self):
        query = [1.0, 0.0]
        # d1 and d2 are identical duplicates
        # d3 is orthogonal and distinct
        docs = [
            ("d1", [0.99, 0.01]),
            ("d2", [0.98, 0.02]),
            ("d3", [0.70, 0.70]),
        ]

        # Greedy picks d1, then d2
        greedy = greedy_top_k(query, docs, top_k=2)
        self.assertEqual(greedy, ["d1", "d2"])

        # MMR with lambda=0.3 prioritizes diversity and picks d1, then d3
        diverse = mmr(query, docs, top_k=2, lambda_=0.3)
        self.assertEqual(diverse, ["d1", "d3"])

    def test_cohere_reranker_fallback(self):
        reranker = CohereReranker()
        docs = [
            "PostgreSQL database query optimization and indexes",
            "Distributed rate limiting with Redis sliding window",
            "Python asyncio event loop"
        ]
        results = reranker.rerank("rate limiting Redis", docs, top_n=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].index, 1)  # Redis rate limiting is most relevant

    def test_reciprocal_rank_fusion_consensus(self):
        sys1 = [
            RankedItem("docA", "Content A", 1),
            RankedItem("docB", "Content B", 2),
        ]
        sys2 = [
            RankedItem("docA", "Content A", 1),
            RankedItem("docC", "Content C", 2),
        ]
        fused = reciprocal_rank_fusion({"Dense": sys1, "BM25": sys2}, k=60, top_n=2)
        self.assertEqual(fused[0].doc_id, "docA")
        self.assertAlmostEqual(fused[0].rrf_score, 2.0 / 61.0, places=5)


if __name__ == "__main__":
    unittest.main()
