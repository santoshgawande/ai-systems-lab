import unittest
import sys, os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-react-loop"))
from react import calculator, read_file, write_file, list_dir

class TestAgentTools(unittest.TestCase):
    def test_calculator_tool(self):
        self.assertEqual(calculator("2 + 2"), "4")
        self.assertEqual(calculator("sqrt(16)"), "4.0")
        self.assertTrue("Error" in calculator("__import__('os').system('ls')"))

    def test_file_io_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            res_write = write_file(test_file, "Hello AI Agent!")
            self.assertIn("Wrote 15 chars", res_write)

            content = read_file(test_file)
            self.assertEqual(content, "Hello AI Agent!")

            dir_listing = list_dir(tmpdir)
            self.assertIn("test.txt", dir_listing)

if __name__ == "__main__":
    unittest.main()
