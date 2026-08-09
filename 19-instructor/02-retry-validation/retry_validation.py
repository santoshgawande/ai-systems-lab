"""
Instructor retry + validation: Pydantic validators that auto-correct via LLM retry.
Shows how field_validator errors become prompts that fix the model's output.
Requires: instructor, anthropic (or openai), pydantic
"""
import os
from typing import Literal

API_KEY_ANTHROPIC = os.environ.get("ANTHROPIC_API_KEY")
API_KEY_OPENAI = os.environ.get("OPENAI_API_KEY")

if API_KEY_ANTHROPIC:
    import instructor
    from anthropic import Anthropic
    _raw_client = Anthropic(api_key=API_KEY_ANTHROPIC)
    client = instructor.from_anthropic(_raw_client, max_retries=3)
    MODEL = "claude-haiku-4-5-20251001"
    PROVIDER = "anthropic"
    LIVE = True
elif API_KEY_OPENAI:
    import instructor
    from openai import OpenAI
    _raw_client = OpenAI(api_key=API_KEY_OPENAI)
    client = instructor.from_openai(_raw_client, max_retries=3)
    MODEL = "gpt-4o-mini"
    PROVIDER = "openai"
    LIVE = True
else:
    LIVE = False

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_OK = True
except ImportError:
    PYDANTIC_OK = False


# ─── Models with validators ───────────────────────────────────────────────────

if PYDANTIC_OK:
    class ProductReview(BaseModel):
        sentiment: Literal["positive", "negative", "neutral", "mixed"]
        score: float = Field(description="Rating 1.0-10.0")
        title: str = Field(description="One sentence summary")
        pros: list[str] = Field(description="Up to 5 specific positives")
        cons: list[str] = Field(description="Up to 5 specific negatives")
        recommend: bool

        @field_validator("score")
        @classmethod
        def score_in_range(cls, v: float) -> float:
            if not 1.0 <= v <= 10.0:
                raise ValueError(f"score must be between 1.0 and 10.0, got {v}")
            return round(v, 1)

        @field_validator("pros", "cons")
        @classmethod
        def max_five_items(cls, v: list[str]) -> list[str]:
            if len(v) > 5:
                raise ValueError(f"maximum 5 items allowed, got {len(v)}")
            return v

        @field_validator("title")
        @classmethod
        def title_is_sentence(cls, v: str) -> str:
            if len(v) > 150:
                raise ValueError(f"title must be under 150 chars, got {len(v)}")
            return v

        @model_validator(mode="after")
        def sentiment_score_consistent(self) -> "ProductReview":
            if self.sentiment == "positive" and self.score < 6.0:
                raise ValueError(
                    f"positive sentiment requires score >= 6.0, but got score={self.score}. "
                    "Either increase the score or change sentiment to neutral/mixed."
                )
            if self.sentiment == "negative" and self.score > 5.0:
                raise ValueError(
                    f"negative sentiment requires score <= 5.0, but got score={self.score}. "
                    "Either decrease the score or change sentiment to neutral/mixed."
                )
            if self.recommend and self.score < 6.0:
                raise ValueError(
                    f"recommend=True requires score >= 6.0, got {self.score}"
                )
            return self

    class EventExtraction(BaseModel):
        name: str
        date: str = Field(description="ISO format YYYY-MM-DD if determinable, else null")
        time: str | None = Field(default=None, description="HH:MM 24h format if present")
        location: str | None = None
        max_attendees: int | None = None
        registration_required: bool
        tags: list[str] = Field(description="At most 5 relevant tags")

        @field_validator("date")
        @classmethod
        def valid_date_format(cls, v: str) -> str:
            import re
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                raise ValueError(f"date must be YYYY-MM-DD format, got {v!r}")
            return v

        @field_validator("max_attendees")
        @classmethod
        def positive_attendees(cls, v: int | None) -> int | None:
            if v is not None and v <= 0:
                raise ValueError(f"max_attendees must be positive, got {v}")
            return v

        @field_validator("tags")
        @classmethod
        def max_five_tags(cls, v: list[str]) -> list[str]:
            if len(v) > 5:
                raise ValueError(f"max 5 tags, got {len(v)}")
            return v


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== INSTRUCTOR RETRY + VALIDATION DEMO ===\n")

if not PYDANTIC_OK:
    print("Install pydantic: pip install pydantic")
elif not LIVE:
    print("Retry validation pattern:\n")
    print("""
from pydantic import BaseModel, field_validator
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic(), max_retries=3)

class Review(BaseModel):
    score: float

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v):
        if not 1 <= v <= 10:
            raise ValueError(f"score must be 1-10, got {v}")
        return v

# If LLM returns score=95, Pydantic raises:
#   ValueError: score must be 1-10, got 95
# Instructor catches this and sends to LLM:
#   "Validation Error: score must be 1-10, got 95. Please fix."
# LLM corrects to score=9.5 on retry.

review = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "This is an amazing product!"}],
    response_model=Review,
)
# review.score is guaranteed to be 1-10
""")
    print("Key validators to add:")
    print("  @field_validator  — validate/transform individual fields")
    print("  @model_validator  — cross-field consistency (score+sentiment match)")
    print("  Field(ge=, le=)   — numeric bounds (Pydantic built-in)")
    print("  Field(max_length=) — string length (Pydantic built-in)")
else:
    def create(text: str, model_class):
        if PROVIDER == "anthropic":
            return client.messages.create(
                model=MODEL, max_tokens=512,
                messages=[{"role": "user", "content": text}],
                response_model=model_class,
            )
        return client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": text}],
            response_model=model_class,
        )

    # Demo 1: Product reviews with cross-field validation
    print("─── Product reviews with score/sentiment consistency ───\n")
    reviews = [
        "This is the worst product I have ever purchased. It is broken, useless, and overpriced.",
        "Absolutely incredible! Changed my life. 10/10 would recommend to everyone.",
        "Decent product, has its good points and bad points, nothing special.",
    ]
    for review_text in reviews:
        result = create(review_text, ProductReview)
        print(f"  Review: {review_text[:60]!r}")
        print(f"  → sentiment={result.sentiment} score={result.score} recommend={result.recommend}")
        print(f"  → title={result.title!r}")
        print()

    # Demo 2: Event extraction with date format enforcement
    print("─── Event extraction with date format enforcement ───\n")
    events = [
        "Join us for the AI Summit on March 15, 2025 at 9:00 AM in San Francisco. Up to 500 attendees. Registration required.",
        "Monthly team lunch — next Friday, no registration needed. Usually around 15-20 people.",
    ]
    for event_text in events:
        result = create(event_text, EventExtraction)
        print(f"  Event: {event_text[:70]!r}")
        print(f"  → name={result.name!r}")
        print(f"  → date={result.date}  time={result.time}")
        print(f"  → location={result.location}  max_attendees={result.max_attendees}")
        print(f"  → registration={result.registration_required}  tags={result.tags}")
        print()

    print("All outputs passed validation — instructor handled any corrections automatically.")
