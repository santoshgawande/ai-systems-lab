import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-pdf-extraction"))
from pdf_extraction import chunk_text

class TestDocumentParsing(unittest.TestCase):
    def test_chunk_text_with_overlap(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        self.assertTrue(len(chunks) >= 2)
        self.assertEqual(chunks[0]["chunk_id"], 0)
        self.assertTrue(chunks[0]["end"] > 0)

if __name__ == "__main__":
    unittest.main()
