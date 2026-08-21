import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-transformers-pipeline"))
from pipeline_demo import TASKS

class TestHuggingFace(unittest.TestCase):
    def test_tasks_catalogue(self):
        task_names = [t["name"] for t in TASKS]
        self.assertIn("Sentiment Analysis", task_names)
        self.assertIn("Named Entity Recognition", task_names)
        self.assertIn("Zero-Shot Classification", task_names)

if __name__ == "__main__":
    unittest.main()
