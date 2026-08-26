import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-prompt-injection"))
from injection import regex_guard, DIRECT_ATTACKS

class TestAISecurity(unittest.TestCase):
    def test_regex_guard_blocks_direct_attacks(self):
        regex_catchable = [a for a in DIRECT_ATTACKS if a["name"] in ("Classic ignore", "Role override", "Subtle", "Encoded")]
        for attack in regex_catchable:
            res = regex_guard(attack["payload"])
            self.assertTrue(res.blocked, f"Failed to block direct attack: {attack['name']}")

    def test_regex_guard_allows_safe_queries(self):
        res = regex_guard("Can you help me design an event-driven architecture using Kafka?")
        self.assertFalse(res.blocked)

if __name__ == "__main__":
    unittest.main()
