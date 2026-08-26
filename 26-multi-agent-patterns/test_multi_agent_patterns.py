import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-orchestrator-subagent"))
from orchestrator import print_concepts

class TestMultiAgentPatterns(unittest.TestCase):
    def test_concepts_callable(self):
        # Verify print_concepts executes without error
        try:
            print_concepts()
            ok = True
        except Exception:
            ok = False
        self.assertTrue(ok)

if __name__ == "__main__":
    unittest.main()
