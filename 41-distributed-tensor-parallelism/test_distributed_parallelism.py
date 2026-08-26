import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-column-row-parallelism"))
sys.path.insert(0, os.path.join(base_dir, "02-pipeline-parallel-1f1b"))

from tensor_parallel import single_gpu_dense_mlp, megatron_tensor_parallel_mlp
from pipeline_1f1b import Pipeline1F1BScheduler


class TestDistributedParallelism(unittest.TestCase):
    def test_megatron_tp_numerical_equality(self):
        X = [1.0, -2.0]
        W1 = [
            [0.5, -0.5, 0.2, 0.8],
            [-0.1, 0.4, -0.3, 0.6]
        ]
        W2 = [
            [0.2, 0.7],
            [-0.4, 0.1],
            [0.9, -0.3],
            [0.1, 0.5]
        ]
        
        single_out = single_gpu_dense_mlp(X, W1, W2)
        tp_out, _ = megatron_tensor_parallel_mlp(X, W1, W2, tp_world_size=2)
        
        for s, t in zip(single_out, tp_out):
            self.assertAlmostEqual(s, t, places=5)

    def test_pipeline_1f1b_scheduler(self):
        scheduler = Pipeline1F1BScheduler(num_stages=4, num_microbatches=8)
        self.assertAlmostEqual(scheduler.bubble_fraction, 3.0 / 8.0)
        
        schedule_stage0 = scheduler.generate_stage_schedule(0)
        # Stage 0: 4 warmup forwards, 4 1F1B pairs (4 F + 4 B), 4 cooldown backwards = 16 events
        self.assertEqual(len(schedule_stage0), 16)


if __name__ == "__main__":
    unittest.main()
