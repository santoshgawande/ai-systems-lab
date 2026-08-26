import unittest
import sys, os

# Add subfolders to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "04-model-router"))
from router import classify_prompt, ROUTING_TABLE

class TestLLMApis(unittest.TestCase):
    def test_model_router_classification(self):
        # Code prompt
        code_model = classify_prompt("Write a python function to compute fibonacci numbers")
        self.assertEqual(code_model, "code")

        # Reasoning prompt
        math_model = classify_prompt("Calculate the equation and solve for x")
        self.assertEqual(math_model, "reasoning")

        # Fast prompt
        fact_model = classify_prompt("What is the capital of France?")
        self.assertEqual(fact_model, "fast")

        # General fallback
        gen_model = classify_prompt("Tell me a creative story about a robot")
        self.assertEqual(gen_model, "general")

    def test_routing_table_structure(self):
        self.assertIn("code", ROUTING_TABLE)
        self.assertIn("reasoning", ROUTING_TABLE)
        self.assertIn("fast", ROUTING_TABLE)
        self.assertIn("general", ROUTING_TABLE)

if __name__ == "__main__":
    unittest.main()
