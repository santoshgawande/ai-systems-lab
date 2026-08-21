import unittest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import QUOTA_SIGNALS, STATUS_CHAR, CHAR_STATUS

class TestAutonomousRunner(unittest.TestCase):
    def test_quota_signals_detection(self):
        self.assertIn("rate limit exceeded", QUOTA_SIGNALS)
        self.assertIn("usage limit reached", QUOTA_SIGNALS)

    def test_status_char_mapping(self):
        self.assertEqual(STATUS_CHAR["pending"], " ")
        self.assertEqual(STATUS_CHAR["done"], "x")
        self.assertEqual(STATUS_CHAR["failed"], "!")
        self.assertEqual(STATUS_CHAR["skipped"], "~")
        self.assertEqual(CHAR_STATUS["x"], "done")

if __name__ == "__main__":
    unittest.main()
