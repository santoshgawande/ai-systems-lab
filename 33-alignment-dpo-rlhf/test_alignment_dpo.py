import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-dpo-loss"))
sys.path.insert(0, os.path.join(base_dir, "02-preference-dataset"))

from dpo import DPOEngine, DPOTrainSample
from preference_pipeline import PreferenceDataPipeline, RawCandidatePair


class TestAlignmentDPO(unittest.TestCase):
    def test_dpo_loss_computation(self):
        engine = DPOEngine(beta=0.1)
        sample = DPOTrainSample(
            prompt="test prompt",
            chosen="good answer",
            rejected="bad answer",
            pi_theta_chosen_logprob=-2.0,
            pi_ref_chosen_logprob=-4.0,   # Delta = +2.0
            pi_theta_rejected_logprob=-6.0,
            pi_ref_rejected_logprob=-4.0  # Delta = -2.0
        )
        loss, rw, rl = engine.compute_sample_loss(sample)
        # rw = 0.1 * 2.0 = 0.2
        # rl = 0.1 * -2.0 = -0.2
        # margin = 0.4
        self.assertAlmostEqual(rw, 0.2)
        self.assertAlmostEqual(rl, -0.2)
        self.assertGreater(rw, rl)
        self.assertGreater(loss, 0.0)

    def test_preference_pipeline_margin_filtering(self):
        pipeline = PreferenceDataPipeline(min_margin=0.2)
        
        # Valid pair with wide margin
        pair1 = RawCandidatePair("p1", "resp1", "resp2", score_a=0.9, score_b=0.3)
        res1 = pipeline.process_pair(pair1)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.chosen, "resp1")
        
        # Pair with narrow margin (should be dropped)
        pair2 = RawCandidatePair("p2", "resp1", "resp2", score_a=0.85, score_b=0.80)
        res2 = pipeline.process_pair(pair2)
        self.assertIsNone(res2)


if __name__ == "__main__":
    unittest.main()
