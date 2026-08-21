#!/usr/bin/env python3
"""
Test runner for all modules in ai-systems-lab.
"""
import unittest
import sys
import os

def run_all():
    suite = unittest.TestSuite()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    import importlib.util
    for root, dirs, files in os.walk(repo_root):
        if any(ignored in root for ignored in [".git", ".system_generated", "venv", "__pycache__"]):
            continue
        for f in sorted(files):
            if f.startswith("test_") and f.endswith(".py") and f != "test_all.py":
                file_path = os.path.join(root, f)
                module_name = os.path.splitext(f)[0] + "_" + os.path.basename(root).replace("-", "_")
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                tests = unittest.defaultTestLoader.loadTestsFromModule(mod)
                suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    run_all()
