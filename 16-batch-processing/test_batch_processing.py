import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "03-async-queue"))
from async_queue import Job

class TestBatchProcessing(unittest.TestCase):
    def test_job_priority_ordering(self):
        j1 = Job(id="job_high", priority=1, prompt="critical task")
        j2 = Job(id="job_low", priority=3, prompt="background task")
        self.assertTrue(j1 < j2)

    def test_job_timing_metrics(self):
        j = Job(id="j1", priority=2, prompt="p")
        j.queued_at = 100.0
        j.started_at = 100.05
        j.completed_at = 100.25
        self.assertAlmostEqual(j.wait_ms, 50.0)
        self.assertAlmostEqual(j.process_ms, 200.0)

if __name__ == "__main__":
    unittest.main()
