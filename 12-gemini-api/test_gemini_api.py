import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-long-context"))
from long_context import build_document

class TestGeminiAPI(unittest.TestCase):
    def test_build_synthetic_document(self):
        doc, facts = build_document(num_sections=10)
        self.assertEqual(len(facts), 10)
        self.assertIn("Section 1:", doc)
        self.assertIn("Section 10:", doc)
        self.assertTrue(facts[0]["code"].startswith("CODE-"))

if __name__ == "__main__":
    unittest.main()
