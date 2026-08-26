import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-multi-token-heads"))
sys.path.insert(0, os.path.join(base_dir, "02-self-speculative-decoding"))

from mtp_engine import MultiTokenPredictionEngine
from self_speculative import MTPSelfSpeculativeDecoder


class TestMultiTokenPrediction(unittest.TestCase):
    def test_mtp_multi_token_heads(self):
        engine = MultiTokenPredictionEngine(num_heads=3)
        logits = [
            {"a": 0.9, "b": 0.1},
            {"b": 0.8, "c": 0.2},
            {"c": 0.95, "d": 0.05}
        ]
        cand = engine.predict_multi_tokens("start", logits)
        self.assertEqual(cand.token_predictions, ["a", "b", "c"])
        
        acc, matches = engine.evaluate_multi_token_loss(cand, ["a", "b", "c", "d"])
        self.assertEqual(matches, 3)
        self.assertAlmostEqual(acc, 1.0)

    def test_self_speculative_decoding(self):
        decoder = MTPSelfSpeculativeDecoder(num_mtp_heads=2)
        proposed = ["tok1", "tok2"]
        backbone_logits = [
            {"tok1": 1.0},
            {"tok2": 1.0}
        ]
        res = decoder.decode_step(proposed, backbone_logits)
        self.assertEqual(res.accepted_tokens, ["tok1", "tok2"])
        self.assertEqual(res.num_accepted, 2)
        self.assertEqual(res.speedup_factor, 2.0)


if __name__ == "__main__":
    unittest.main()
