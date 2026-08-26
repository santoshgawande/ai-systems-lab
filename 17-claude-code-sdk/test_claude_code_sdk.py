import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "02-hooks"))
from hooks import settings_json, BASH_GUARD_SCRIPT, WRITE_FORMAT_SCRIPT

class TestClaudeCodeSDK(unittest.TestCase):
    def test_settings_json_structure(self):
        settings = settings_json("/fake/path")
        self.assertIn("hooks", settings)
        self.assertIn("PreToolUse", settings["hooks"])
        self.assertIn("PostToolUse", settings["hooks"])

    def test_bash_guard_script_blocks(self):
        self.assertIn("rm -rf", BASH_GUARD_SCRIPT)
        self.assertIn("chmod 777", BASH_GUARD_SCRIPT)
        self.assertIn("exit 2", BASH_GUARD_SCRIPT)

if __name__ == "__main__":
    unittest.main()
