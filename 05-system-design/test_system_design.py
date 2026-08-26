import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-cost-optimization"))
from cost import token_estimate, cost_usd, classify_tier, route

class TestSystemDesign(unittest.TestCase):
    def test_token_estimation(self):
        self.assertEqual(token_estimate(""), 1)
        self.assertEqual(token_estimate("Hello world! 1234"), 4)

    def test_cost_calculation(self):
        # phi4: 0.0001 / 1k input, 0.0002 / 1k output
        c = cost_usd("phi4", input_t=1000, output_t=1000)
        self.assertAlmostEqual(c, 0.0003)

    def test_tier_classification_and_routing(self):
        self.assertEqual(classify_tier("What is the capital of Japan?"), "cheap")
        self.assertEqual(route("cheap"), "phi4")

        self.assertEqual(classify_tier("Analyze the trade-offs between SQL and NoSQL for financial ledgers"), "premium")
        self.assertEqual(route("premium"), "llama3.3:70b")

if __name__ == "__main__":
    unittest.main()
