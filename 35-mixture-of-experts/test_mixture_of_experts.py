import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-topk-gating"))
sys.path.insert(0, os.path.join(base_dir, "02-expert-load-balancing"))

from moe_gating import MoETopKGatingRouter
from load_balancer import MoELoadBalancer


class TestMixtureOfExperts(unittest.TestCase):
    def test_moe_topk_gating_and_normalization(self):
        router = MoETopKGatingRouter(num_experts=4, top_k=2, has_shared_expert=True)
        logits = [2.0, 5.0, 1.0, 0.5]
        
        dispatch = router.route_token(logits, token_idx=0)
        # Should pick index 1 (5.0) and index 0 (2.0)
        self.assertEqual(dispatch.selected_expert_ids, [1, 0])
        self.assertEqual(len(dispatch.expert_weights), 2)
        self.assertAlmostEqual(sum(dispatch.expert_weights), 1.0)
        self.assertTrue(dispatch.shared_expert_active)

    def test_moe_load_balancer_auxiliary_loss(self):
        balancer = balancer_mod_instance = MoELoadBalancer(num_experts=2, alpha=0.01)
        
        # Balanced: 50% expert 0, 50% expert 1
        probs = [[0.5, 0.5], [0.5, 0.5]]
        indices = [[0], [1]]
        metrics_balanced = balancer.compute_auxiliary_loss(probs, indices)
        
        # Collapsed: 100% expert 0
        probs_coll = [[1.0, 0.0], [1.0, 0.0]]
        indices_coll = [[0], [0]]
        metrics_collapsed = balancer.compute_auxiliary_loss(probs_coll, indices_coll)
        
        self.assertGreater(metrics_collapsed.auxiliary_loss, metrics_balanced.auxiliary_loss)
        self.assertTrue(metrics_balanced.is_balanced)


if __name__ == "__main__":
    unittest.main()
