import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-prompt-caching"))
from caching import cost_usd

class TestClaudeAPI(unittest.TestCase):
    def test_cost_usd_without_cache(self):
        # 1M input ($3) + 1M output ($15) = $18
        c = cost_usd(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(c, 18.0)

    def test_cost_usd_with_cache_read_discount(self):
        # 1M cache read ($0.30) vs 1M standard input ($3.00) -> 90% savings
        c_cached = cost_usd(input_tokens=0, output_tokens=0, cache_read=1_000_000)
        self.assertAlmostEqual(c_cached, 0.30)

        c_write = cost_usd(input_tokens=0, output_tokens=0, cache_write=1_000_000)
        self.assertAlmostEqual(c_write, 3.75)

if __name__ == "__main__":
    unittest.main()
