import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-sliding-window"))
from sliding_window import count_tokens_approx, apply_sliding_window, conversation_tokens

class TestContextManagement(unittest.TestCase):
    def test_token_counting(self):
        self.assertEqual(count_tokens_approx("Hello world! 1234"), 4)

    def test_sliding_window_eviction(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question 1 that is fairly long and detailed."},
            {"role": "assistant", "content": "Answer 1 with lots of explanations and text."},
            {"role": "user", "content": "Question 2."},
            {"role": "assistant", "content": "Answer 2."},
        ]
        total = conversation_tokens(msgs)

        # Set limit lower than total -> oldest non-system evicted
        trimmed, evicted = apply_sliding_window(msgs, max_tokens=total - 20)
        self.assertTrue(evicted >= 1)
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[-1]["content"], "Answer 2.")

if __name__ == "__main__":
    unittest.main()
