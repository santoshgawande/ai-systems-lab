import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-dynamic-fact-extraction"))
sys.path.insert(0, os.path.join(base_dir, "02-memory-consolidation"))

from fact_extractor import DynamicFactExtractor, AtomicMemoryFact
from memory_graph import MemoryGraphConsolidator


class TestAgenticMemoryGraph(unittest.TestCase):
    def test_fact_extraction(self):
        extractor = DynamicFactExtractor()
        facts = extractor.extract_facts_from_turn("I live in Bangalore and I write in Python.")
        self.assertEqual(len(facts), 2)
        
        preds = {f.predicate: f.object_value for f in facts}
        self.assertEqual(preds["lives_in"], "Bangalore")
        self.assertEqual(preds["prefers_language"], "Python")

    def test_memory_consolidation_and_supersession(self):
        consolidator = MemoryGraphConsolidator()
        
        f1 = AtomicMemoryFact("f1", "User", "lives_in", "Delhi", 0.9, 1000.0, "source1")
        f2 = AtomicMemoryFact("f2", "User", "lives_in", "Pune", 0.95, 2000.0, "source2")
        
        consolidator.insert_fact(f1)
        self.assertEqual(len(consolidator.query_user_profile("User")), 1)
        
        # Insert newer conflicting fact
        consolidator.insert_fact(f2)
        profile = consolidator.query_user_profile("User")
        self.assertEqual(len(profile), 1)
        self.assertEqual(profile[0]["value"], "Pune")
        self.assertEqual(len(consolidator.archived_history), 1)


if __name__ == "__main__":
    unittest.main()
