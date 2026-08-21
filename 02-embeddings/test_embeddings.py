import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-what-are-embeddings"))
from embed import cosine

class TestEmbeddings(unittest.TestCase):
    def test_cosine_similarity_identical_vectors(self):
        v1 = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine(v1, v1), 1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        self.assertAlmostEqual(cosine(v1, v2), 0.0)

    def test_cosine_similarity_opposite_vectors(self):
        v1 = [1.0, 2.0]
        v2 = [-1.0, -2.0]
        self.assertAlmostEqual(cosine(v1, v2), -1.0)

if __name__ == "__main__":
    unittest.main()
