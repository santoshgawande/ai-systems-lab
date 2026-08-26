import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-speculative-sampling"))
sys.path.insert(0, os.path.join(base_dir, "02-medusa-heads"))

from speculative_sampling import SpeculativeSamplingEngine, compute_residual_distribution
from medusa import MedusaEngine


class TestSpeculativeDecoding(unittest.TestCase):
    def test_speculative_sampling_perfect_acceptance(self):
        engine = SpeculativeSamplingEngine(gamma=3, seed=42)
        draft_tokens = ["hello", "world", "!"]
        # Perfect alignment: draft distribution == target distribution
        p_dists = [{"hello": 1.0}, {"world": 1.0}, {"!": 1.0}]
        q_dists = [{"hello": 1.0}, {"world": 1.0}, {"!": 1.0}, {".": 1.0}]
        
        accepted, count = engine.verify_draft_sequence(draft_tokens, p_dists, q_dists)
        self.assertEqual(count, 3)
        self.assertEqual(accepted[:3], ["hello", "world", "!"])
        self.assertEqual(len(accepted), 4)  # 3 draft + 1 bonus target token
        self.assertAlmostEqual(engine.acceptance_rate, 1.0)

    def test_residual_distribution_calculation(self):
        target_dist = {"apple": 0.6, "banana": 0.4}
        draft_dist = {"apple": 0.8, "banana": 0.2}
        
        residual = compute_residual_distribution(target_dist, draft_dist)
        # apple: max(0, 0.6 - 0.8) = 0.0
        # banana: max(0, 0.4 - 0.2) = 0.2
        self.assertAlmostEqual(residual["apple"], 0.0)
        self.assertAlmostEqual(residual["banana"], 1.0)

    def test_medusa_candidate_tree_and_verification(self):
        engine = MedusaEngine(num_heads=2, topk_per_head=2)
        head_preds = [
            [("token_a", 0.7), ("token_b", 0.3)],
            [("token_c", 0.8), ("token_d", 0.2)]
        ]
        candidates = engine.generate_candidate_tree("root", head_preds)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0].depth, 2)
        
        # Verify longest prefix match
        ground_truth = ["token_a", "token_c", "extra"]
        accepted, count = engine.verify_longest_prefix(ground_truth, candidates)
        self.assertEqual(accepted, ["token_a", "token_c"])
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
