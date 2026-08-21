import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-mmr"))
from mmr import mmr, cosine, greedy_top_k

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

if __name__ == "__main__":
    unittest.main()
