import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-self-critique"))
from self_critique import PRINCIPLES, TEST_CASES

class TestConstitutionalAI(unittest.TestCase):
    def test_principles_definition(self):
        self.assertTrue(len(PRINCIPLES) >= 5)
        self.assertTrue(any("harm" in p for p in PRINCIPLES))
        self.assertTrue(any("privacy" in p for p in PRINCIPLES))

    def test_test_cases_structure(self):
        self.assertTrue(len(TEST_CASES) >= 2)
        self.assertIn("name", TEST_CASES[0])
        self.assertIn("request", TEST_CASES[0])
        self.assertIn("principles", TEST_CASES[0])

if __name__ == "__main__":
    unittest.main()
