import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-unit-evals"))
from eval import Eval, extract_json

class TestEvals(unittest.TestCase):
    def test_extract_json_util(self):
        self.assertEqual(extract_json('{"name": "Bob", "age": 30}'), {"name": "Bob", "age": 30})
        self.assertEqual(extract_json('```json\n{"score": 100}\n```'), {"score": 100})
        self.assertIsNone(extract_json("Not a json string"))

    def test_eval_assertion_runner(self):
        ev = Eval(
            name="sentiment_pos",
            system="Classify sentiment",
            input="Great service!",
            check=lambda r: "POSITIVE" in r.upper(),
            expected="POSITIVE"
        )
        self.assertTrue(ev.check("POSITIVE"))
        self.assertFalse(ev.check("NEGATIVE"))

if __name__ == "__main__":
    unittest.main()
