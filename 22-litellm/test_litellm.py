import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-routing"))
import routing

class TestLiteLLM(unittest.TestCase):
    def test_routing_module_attributes(self):
        self.assertTrue(hasattr(routing, "demo_concepts"))
        self.assertTrue(hasattr(routing, "run_router_demo"))

if __name__ == "__main__":
    unittest.main()
