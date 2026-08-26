import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mini-eval-framework"))
from eval_framework import Eval, EvalResult, Status

class TestBuildYourOwn(unittest.TestCase):
    def test_eval_definition_and_custom_check(self):
        ev = Eval(
            id="json-output-01",
            suite="formatting",
            description="Must output valid json",
            system="Always output JSON",
            prompt="Give me 2 colors",
            check=lambda text: text.strip().startswith("{") and text.strip().endswith("}")
        )
        self.assertEqual(ev.id, "json-output-01")
        self.assertTrue(ev.check('{"colors": ["red", "blue"]}'))
        self.assertFalse(ev.check("Here are the colors: red, blue"))

    def test_eval_result_dataclass(self):
        ev = Eval(
            id="length-01",
            suite="conciseness",
            description="Under 50 chars",
            system="",
            prompt="Summarize today",
            check=lambda text: len(text) < 50
        )
        res = EvalResult(
            eval=ev,
            response="Short summary.",
            status=Status.PASS,
            latency_ms=120.5
        )
        self.assertEqual(res.status, Status.PASS)
        self.assertEqual(res.latency_ms, 120.5)

if __name__ == "__main__":
    unittest.main()
