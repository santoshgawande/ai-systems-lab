import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-knowledge-graph-extraction"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-community-detection-hierarchical-rag"))

from kg_extractor import KnowledgeGraphExtractor, KnowledgeGraph
from graph_rag import HierarchicalGraphRAGEngine


class TestGraphRAG(unittest.TestCase):

    def test_kg_extraction_and_neighbors(self):
        text = "Redis stores cached tokens\nPostgreSQL stores accounts"
        kg = KnowledgeGraphExtractor.extract_from_text(text)
        self.assertGreater(len(kg.entities), 2)
        self.assertGreater(len(kg.relationships), 2)
        neighbors = kg.get_neighbors("RateLimiter")
        self.assertGreater(len(neighbors), 0)

    def test_hierarchical_graph_rag_local_search(self):
        engine = HierarchicalGraphRAGEngine()
        res = engine.local_search("RateLimiter")
        self.assertEqual(res.mode, "LOCAL_SEARCH")
        self.assertIn("RateLimiter", res.answer)
        self.assertGreater(len(res.context_sources), 0)

    def test_hierarchical_graph_rag_global_search(self):
        engine = HierarchicalGraphRAGEngine()
        res = engine.global_search("What are all domains?")
        self.assertEqual(res.mode, "GLOBAL_SEARCH")
        self.assertIn("Traffic Ingress", res.answer)
        self.assertIn("Transactional Order", res.answer)
        self.assertEqual(len(res.context_sources), 2)


if __name__ == "__main__":
    unittest.main()
