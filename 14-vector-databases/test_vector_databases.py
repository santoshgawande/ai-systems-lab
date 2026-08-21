import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "03-metadata-filtering"))
from filtering import DOCS, BODY_MAP

class TestVectorDatabases(unittest.TestCase):
    def test_docs_dataset_structure(self):
        self.assertEqual(len(DOCS), 10)
        self.assertTrue(all("category" in d and "product" in d for d in DOCS))
        self.assertTrue(all(d["id"] in BODY_MAP for d in DOCS))

    def test_filter_matching_logic(self):
        # Database category count
        db_docs = [d for d in DOCS if d["category"] == "database"]
        self.assertEqual(len(db_docs), 4)

        # Published only count
        pub_docs = [d for d in DOCS if d["published"]]
        self.assertEqual(len(pub_docs), 9)

if __name__ == "__main__":
    unittest.main()
