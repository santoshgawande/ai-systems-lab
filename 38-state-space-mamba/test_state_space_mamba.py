import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-selective-scan-ssm"))
sys.path.insert(0, os.path.join(base_dir, "02-linear-time-inference"))

from selective_scan import SelectiveSSMBlock
from mamba_inference import MambaInferenceEngine


class TestStateSpaceMamba(unittest.TestCase):
    def test_ssm_step_and_scan(self):
        ssm = SelectiveSSMBlock(state_dim=4)
        results = ssm.scan_sequence([1.0, -1.0, 2.0])
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0].hidden_state), 4)
        self.assertGreater(results[0].delta_t, 0.0)

    def test_mamba_constant_memory_inference(self):
        engine = MambaInferenceEngine(state_dim=16, transformer_kv_dim=64)
        stats = engine.simulate_generation(num_tokens=10)
        
        # Mamba memory must remain constant across all steps
        mamba_mems = stats["mamba_memory_per_step"]
        self.assertTrue(all(m == 16 for m in mamba_mems))
        
        # Transformer memory must strictly increase
        trans_mems = stats["transformer_memory_per_step"]
        self.assertEqual(trans_mems[-1], 640)
        self.assertGreater(trans_mems[-1], trans_mems[0])


if __name__ == "__main__":
    unittest.main()
