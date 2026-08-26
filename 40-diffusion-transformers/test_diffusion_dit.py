import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-dit-patchification"))
sys.path.insert(0, os.path.join(base_dir, "02-classifier-free-guidance"))

from dit_transformer import DiffusionTransformerEngine
from cfg_sampling import ClassifierFreeGuidanceEngine


class TestDiffusionDiT(unittest.TestCase):
    def test_dit_patchification_dimensions(self):
        engine = DiffusionTransformerEngine(patch_size=2)
        # 4x4 image with 1 channel
        grid = [[[float(i+j)] for j in range(4)] for i in range(4)]
        
        seq = engine.patchify_latent(grid)
        # (4/2) * (4/2) = 4 patches
        self.assertEqual(len(seq.patches), 4)
        # Each patch is 2*2*1 = 4 floats
        self.assertEqual(len(seq.patches[0]), 4)

    def test_cfg_guidance_vector_scaling(self):
        engine = ClassifierFreeGuidanceEngine(guidance_scale=2.0)
        cond = [2.0, 4.0]
        uncond = [1.0, 1.0]
        
        # v = 1 + 2 * (2 - 1) = 3
        # v = 1 + 2 * (4 - 1) = 7
        guided = engine.compute_cfg_vector(cond, uncond)
        self.assertEqual(guided, [3.0, 7.0])


if __name__ == "__main__":
    unittest.main()
