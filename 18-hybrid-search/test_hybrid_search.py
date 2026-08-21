import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-bm25"))
from bm25 import SimpleBM25, DOCS, get_text

class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        texts = [get_text(d) for d in DOCS]
        self.bm25 = SimpleBM25(texts)

    def test_bm25_exact_code_identifier_search(self):
        # Query for CustomerNotFoundException
        results = self.bm25.search("CustomerNotFoundException", top_k=1)
        self.assertEqual(len(results), 1)
        best_idx, score = results[0]
        self.assertEqual(DOCS[best_idx]["title"], "Error: CustomerNotFoundException")
        self.assertTrue(score > 0)

    def test_bm25_database_index_search(self):
        results = self.bm25.search("PostgreSQL indexes", top_k=3)
        top_titles = [DOCS[idx]["title"] for idx, _ in results]
        self.assertIn("PostgreSQL Index Types", top_titles)

if __name__ == "__main__":
    unittest.main()
