import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-when-to-fine-tune"))
from decision import DECISIONS, QUESTIONS

class TestFineTuningDecision(unittest.TestCase):
    def test_decision_options_present(self):
        self.assertIn("PROMPTING", DECISIONS)
        self.assertIn("RAG", DECISIONS)
        self.assertIn("FINE_TUNE", DECISIONS)
        self.assertIn("COLLECT_DATA", DECISIONS)

    def test_decision_questions_coverage(self):
        q_ids = [q["id"] for q in QUESTIONS]
        self.assertIn("prompt_works", q_ids)
        self.assertIn("needs_knowledge", q_ids)
        self.assertIn("has_data", q_ids)

if __name__ == "__main__":
    unittest.main()
