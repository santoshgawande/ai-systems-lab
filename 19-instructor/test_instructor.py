import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-basics"))
from instructor_basics import ContactInfo, TaskItem, MeetingNotes, SentimentAnalysis

class TestInstructor(unittest.TestCase):
    def test_contact_info_validation(self):
        c = ContactInfo(name="Alice", email="alice@example.com")
        self.assertEqual(c.name, "Alice")
        self.assertEqual(c.email, "alice@example.com")
        self.assertIsNone(c.phone)

    def test_meeting_notes_nested_model(self):
        item = TaskItem(title="Fix bug", priority="high", assignee="Bob")
        notes = MeetingNotes(
            title="Sprint Planning",
            attendees=["Alice", "Bob"],
            decisions=["Ship v1"],
            action_items=[item]
        )
        self.assertEqual(len(notes.action_items), 1)
        self.assertEqual(notes.action_items[0].priority, "high")

if __name__ == "__main__":
    unittest.main()
