import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-latent-kv-compression"))
sys.path.insert(0, os.path.join(base_dir, "02-matrix-absorption"))

from mla_compression import MultiHeadLatentAttentionEngine
from matrix_absorption import standard_unabsorbed_attention_score, absorbed_attention_score


class TestMLAAttention(unittest.TestCase):
    def test_mla_compression_and_savings(self):
        engine = MultiHeadLatentAttentionEngine(
            hidden_dim=512,
            num_heads=8,
            head_dim=64,
            latent_kv_dim=64,
            rope_dim=16
        )
        # MHA: 2 * 8 * 64 = 1024 floats
        # MLA: 64 + 16 = 80 floats
        self.assertEqual(engine.mha_kv_per_token, 1024)
        self.assertEqual(engine.mla_kv_per_token, 80)
        
        savings = engine.compute_memory_savings(sequence_length=1024, batch_size=4)
        self.assertGreater(savings["savings_percentage"], 90.0)
        self.assertGreater(savings["compression_ratio"], 10.0)

    def test_matrix_absorption_exact_equality(self):
        q = [1.0, 2.0, -1.0]
        c_kv = [0.5, -0.5]
        W_UK = [
            [0.2, -0.1],
            [0.4, 0.3],
            [-0.2, 0.5]
        ]
        
        score_unabs = standard_unabsorbed_attention_score(q, c_kv, W_UK)
        score_abs = absorbed_attention_score(q, c_kv, W_UK)
        self.assertAlmostEqual(score_unabs, score_abs, places=6)


if __name__ == "__main__":
    unittest.main()
