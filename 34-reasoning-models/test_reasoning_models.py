import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-test-time-compute"))
sys.path.insert(0, os.path.join(base_dir, "02-process-reward-verifier"))

from budget_forcing import ThinkingBudgetManager
from prm_verifier import ProcessRewardVerifier, ReasoningTrajectory


class TestReasoningModels(unittest.TestCase):
    def test_thinking_budget_parsing(self):
        manager = ThinkingBudgetManager(min_thought_tokens=5)
        raw = "<think>Step 1. Step 2. Wait, let me double check.</think>Final Answer: 42"
        
        parsed = manager.parse_reasoning_trace(raw)
        self.assertEqual(parsed.final_answer, "Final Answer: 42")
        self.assertGreater(parsed.thought_token_count, 0)
        self.assertGreater(parsed.num_self_corrections, 0)
        self.assertTrue(parsed.budget_met)

    def test_prm_step_verification_and_error_detection(self):
        verifier = ProcessRewardVerifier()
        
        traj_flawed = ReasoningTrajectory(
            "flawed",
            ["Step 1: x = 10", "Step 2: divide by zero: x / 0", "Step 3: done"],
            "done"
        )
        _, _, first_err = verifier.evaluate_trajectory(traj_flawed)
        self.assertEqual(first_err, 2)
        
        traj_valid = ReasoningTrajectory(
            "valid",
            ["Step 1: x = 10", "Step 2: x = x + 5", "Step 3: done"],
            "15"
        )
        _, _, no_err = verifier.evaluate_trajectory(traj_valid)
        self.assertIsNone(no_err)
        
        best, _ = verifier.select_best_trajectory([traj_flawed, traj_valid])
        self.assertEqual(best.trajectory_id, "valid")


if __name__ == "__main__":
    unittest.main()
