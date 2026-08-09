"""
Instructor basics: extract typed Pydantic objects from LLM responses.
Shows single extraction, nested models, lists, and optional fields.
Requires: instructor, anthropic (or openai), pydantic
"""
import os
from typing import Literal

API_KEY_ANTHROPIC = os.environ.get("ANTHROPIC_API_KEY")
API_KEY_OPENAI = os.environ.get("OPENAI_API_KEY")

if API_KEY_ANTHROPIC:
    import instructor
    from anthropic import Anthropic
    client = instructor.from_anthropic(Anthropic(api_key=API_KEY_ANTHROPIC))
    MODEL = "claude-haiku-4-5-20251001"
    PROVIDER = "anthropic"
    LIVE = True
elif API_KEY_OPENAI:
    import instructor
    from openai import OpenAI
    client = instructor.from_openai(OpenAI(api_key=API_KEY_OPENAI))
    MODEL = "gpt-4o-mini"
    PROVIDER = "openai"
    LIVE = True
else:
    LIVE = False
    print("No API key set. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.\n")

try:
    from pydantic import BaseModel, Field
    PYDANTIC_OK = True
except ImportError:
    PYDANTIC_OK = False
    print("pydantic not installed. pip install pydantic\n")

# ─── Schema definitions ───────────────────────────────────────────────────────

if PYDANTIC_OK:
    class ContactInfo(BaseModel):
        name: str
        email: str | None = None
        phone: str | None = None
        company: str | None = None
        role: str | None = None

    class TaskItem(BaseModel):
        title: str
        priority: Literal["high", "medium", "low"]
        assignee: str | None = None
        due: str | None = None

    class MeetingNotes(BaseModel):
        title: str
        date: str | None = None
        attendees: list[str]
        decisions: list[str]
        action_items: list[TaskItem]
        next_meeting: str | None = None

    class SentimentAnalysis(BaseModel):
        sentiment: Literal["positive", "negative", "neutral", "mixed"]
        score: float = Field(ge=1.0, le=10.0, description="Sentiment score 1-10")
        key_phrases: list[str]
        summary: str

    class ResumeData(BaseModel):
        name: str
        email: str | None = None
        years_experience: int | None = None
        skills: list[str]
        current_role: str | None = None
        education: str | None = None


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== INSTRUCTOR BASICS DEMO ===\n")

if not PYDANTIC_OK:
    print("Install pydantic to continue: pip install pydantic")
elif not LIVE:
    print("Instructor API shape:\n")
    print("""
import instructor
from anthropic import Anthropic
from pydantic import BaseModel

client = instructor.from_anthropic(Anthropic())

class User(BaseModel):
    name: str
    age: int
    email: str | None = None

# Returns a User object — ALWAYS valid (instructor retries on failure)
user = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "Extract: John Smith, age 30, john@example.com"}],
    response_model=User,
)
print(user.name, user.age)   # John Smith, 30 — type-safe, no json.loads()

# OpenAI works exactly the same
from openai import OpenAI
client = instructor.from_openai(OpenAI())
user = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract: Jane Doe, 25, jane@example.com"}],
    response_model=User,
)
""")
    print("Supported providers: Anthropic, OpenAI, Gemini, Cohere, Mistral, Ollama")
else:
    def extract(text: str, model_class, label: str) -> object:
        return client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": text}],
            response_model=model_class,
        ) if PROVIDER == "anthropic" else client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": text}],
            response_model=model_class,
        )

    # Demo 1: Contact extraction
    print("─── Contact extraction ───\n")
    texts = [
        "Hi, I'm Dr. Sarah Chen, Head of AI at DeepMind. Reach me at sarah@deepmind.ai or +1-650-555-0142.",
        "Bob from sales here. My email is bob.jones@company.com",
        "Prof. Martinez — no email please, just call the department.",
    ]
    for text in texts:
        result = extract(text, ContactInfo, "ContactInfo")
        print(f"  Input: {text[:60]!r}")
        print(f"  → {result.model_dump(exclude_none=True)}\n")

    # Demo 2: Sentiment analysis with score
    print("─── Sentiment analysis ───\n")
    reviews = [
        "Best software I've ever used. Saves me 2 hours daily. The UI is intuitive and the support team is incredible.",
        "Crashes every 30 minutes. Lost my work 3 times this week. Absolute garbage.",
    ]
    for review in reviews:
        result = extract(review, SentimentAnalysis, "SentimentAnalysis")
        print(f"  {review[:60]!r}")
        print(f"  → sentiment={result.sentiment} score={result.score}/10")
        print(f"  → phrases={result.key_phrases}")
        print(f"  → {result.summary}\n")

    # Demo 3: Meeting notes with nested structure
    print("─── Meeting notes with nested action items ───\n")
    notes = """
    Sprint review meeting on Thursday.
    Attendees: Alice (PM), Bob (eng), Carol (design).
    We decided to launch the feature next Monday.
    Action items: Alice will write the release notes by Friday,
    Bob needs to fix the critical login bug ASAP (high priority),
    Carol to update the design system documentation (low priority).
    Next meeting: Monday 10am.
    """
    result = extract(notes, MeetingNotes, "MeetingNotes")
    print(f"  Meeting: {result.title}")
    print(f"  Attendees: {result.attendees}")
    print(f"  Decisions: {result.decisions}")
    print(f"  Action items:")
    for item in result.action_items:
        print(f"    [{item.priority.upper()}] {item.title} → {item.assignee} (due: {item.due})")
    print(f"  Next meeting: {result.next_meeting}\n")
