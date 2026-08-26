import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-online-softmax"))
sys.path.insert(0, os.path.join(base_dir, "02-sram-block-tiling"))

from online_softmax import standard_softmax, online_softmax, chunked_online_softmax
from flash_attention import standard_attention, flash_attention_tiled


class TestFlashAttention(unittest.TestCase):
    def test_online_softmax_equivalence(self):
        vec = [1.2, 5.5, 3.1, 8.9, 0.4]
        std_p = standard_softmax(vec)
        on_p, _, _ = online_softmax(vec)
        
        for sp, op in zip(std_p, on_p):
            self.assertAlmostEqual(sp, op, places=5)
            
        # Test chunked
        chunk_p, _, _ = chunked_online_softmax([[1.2, 5.5], [3.1, 8.9, 0.4]])
        for sp, cp in zip(std_p, chunk_p):
            self.assertAlmostEqual(sp, cp, places=5)

    def test_flash_attention_tiled_output(self):
        Q = [[1.0, 0.5], [0.2, 0.8]]
        K = [[0.9, 0.4], [0.3, 0.7]]
        V = [[5.0, 1.0], [2.0, 6.0]]
        
        O_std = standard_attention(Q, K, V)
        O_flash = flash_attention_tiled(Q, K, V, B_r=1, B_c=1)
        
        for i in range(len(Q)):
            for j in range(len(Q[0])):
                self.assertAlmostEqual(O_std[i][j], O_flash[i][j], places=4)


if __name__ == "__main__":
    unittest.main()
