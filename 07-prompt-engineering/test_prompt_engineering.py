import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "03-structured-output"))
from structured import extract_json

class TestPromptEngineering(unittest.TestCase):
    def test_extract_json_direct(self):
        text = '{"name": "Alice", "score": 95}'
        parsed = extract_json(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["name"], "Alice")
        self.assertEqual(parsed["score"], 95)

    def test_extract_json_from_markdown_fence(self):
        text = "Here is the extracted JSON:\n```json\n{\n  \"action\": \"search\",\n  \"query\": \"ai models\"\n}\n```\nHope that helps!"
        parsed = extract_json(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "search")
        self.assertEqual(parsed["query"], "ai models")

    def test_extract_json_from_mixed_prose(self):
        text = "Based on the text, the result is {\"status\": \"ok\", \"count\": 3} as expected."
        parsed = extract_json(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["count"], 3)

if __name__ == "__main__":
    unittest.main()
