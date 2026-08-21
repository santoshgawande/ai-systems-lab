import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-chunking"))
from chunk import chunk_fixed, chunk_sentence, chunk_paragraph, chunk_recursive

class TestRAGChunking(unittest.TestCase):
    def setUp(self):
        self.sample = (
            "Paragraph one is about machine learning and data science. It covers algorithms.\n\n"
            "Paragraph two discusses large language models and transformers. Attention is all you need.\n\n"
            "Paragraph three covers vector embeddings and similarity search."
        )

    def test_chunk_fixed(self):
        chunks = chunk_fixed(self.sample, size=50, overlap=10)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(c) <= 50 for c in chunks))

    def test_chunk_sentence(self):
        chunks = chunk_sentence(self.sample, per_chunk=2)
        self.assertTrue(len(chunks) >= 2)

    def test_chunk_paragraph(self):
        chunks = chunk_paragraph(self.sample)
        self.assertEqual(len(chunks), 3)
        self.assertTrue("Paragraph one" in chunks[0])
        self.assertTrue("Paragraph two" in chunks[1])
        self.assertTrue("Paragraph three" in chunks[2])

    def test_chunk_recursive(self):
        chunks = chunk_recursive(self.sample, max_size=100, overlap=20)
        self.assertTrue(len(chunks) >= 3)
        self.assertTrue(all(len(c) <= 120 for c in chunks))

if __name__ == "__main__":
    unittest.main()
